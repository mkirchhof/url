"""
Contains methods to turn each model into a model that also returns uncertainty estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import ResNetDropout
from timm.models.vision_transformer import VisionTransformerDropout, Block
import numpy as np
from timm.data.dataset import EmbeddingTensor

class ModelWrapper(nn.Module):
    """
    This module takes a model as input and then performs all possible functions on the model's functions.
    Children of the ModelWrapper class will be allowed to overwrite these functions.
    This "dirty" implementation is because we do not know which class our given model will have,
    so we cannot just be a subclass of it.
    If this does not work, we could go even more dirty and replace the forward functions at inference time:
    https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes
        if hasattr(model, "drop_rate"):
            self.drop_rate = model.drop_rate
        self.grad_checkpointing = model.grad_checkpointing
        self.num_features = model.num_features

    @torch.jit.ignore
    def group_matcher(self, *args, **kwargs):
        return self.model.group_matcher(*args, **kwargs)

    @torch.jit.ignore
    def set_grad_checkpointing(self, *args, **kwargs):
        res = self.model.set_grad_checkpointing(*args, **kwargs)
        self.grad_checkpointing = self.model.grad_checkpointing
        return res

    @torch.jit.ignore
    def get_classifier(self, *args, **kwargs):
        return self.model.get_classifier(*args, **kwargs)

    def reset_classifier(self, *args, **kwargs):
        return self.model.reset_classifier(*args, **kwargs)

    @staticmethod
    def _is_processed_embedding(x):
        return isinstance(x, EmbeddingTensor)

    def model_forward_features(self, x, **kwargs):
        if self._is_processed_embedding(x):
            return x
        else:
            return self.model.forward_features(x, **kwargs)

    def model_forward_head(self, x, **kwargs):
        if self._is_processed_embedding(x):
            return x
        else:
            return self.model.forward_head(x, **kwargs)

    def forward_features(self, *args, **kwargs):
        return self.model.forward_features(*args, **kwargs)

    def forward_head(self, *args, **kwargs):
        return self.model.forward_head(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


class ShallowEnsembleClassifier(nn.Module):
    def __init__(
        self, num_heads, num_features, num_classes
    ) -> None:
        super().__init__()
        self.shallow_classifiers = nn.Linear(num_features, num_classes * num_heads)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.shallow_classifiers(x).reshape(-1, self.num_heads, self.num_classes)  # [B, N, C]
        return logits.transpose(0, 1)


class ShallowEnsembleWrapper(ModelWrapper):
    """
    This module takes a model as input and creates a shallow ensemble from it.
    """

    def __init__(self, model, num_heads) -> None:
        super().__init__(model)
        # WARNING: self.num_features fails with catavgmax
        # There, pooling doubles feature dims so this
        # ensemble head results in a shape error
        self.classifier = ShallowEnsembleClassifier(
            num_heads, self.num_features, self.num_classes
        )
        self.num_heads = num_heads

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_heads=None, *args, **kwargs):
        if num_heads is None:
            num_heads = self.num_heads
        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        self.num_classes = self.model.num_classes
        self.classifier = ShallowEnsembleClassifier(
            num_heads, self.num_features, self.num_classes
        )

    def forward_features(self, *args, **kwargs):
        # No change via ensembling
        return self.model_forward_features(*args, **kwargs)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        x = self.model_forward_head(x, pre_logits=True)

        # Optionally apply `self.classifier`
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class UncertaintyWrapper(ModelWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.unc_scaler = 1.0

    def initialize_avg_uncertainty(self, loader_train, target_avg_unc, n_batches=10):
        # Find out which uncertainty the model currently predicts on average
        avg_unc = 0.0
        data_iter = loader_train.__iter__()
        prev_state = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                input, _ = data_iter.__next__()
                _, unc, _ = self(input)
                avg_unc += unc.mean().detach().cpu().item() / n_batches
        
        self.train(prev_state)

        # Match our unc_scaler to meet the target_avg_unc
        self.unc_scaler = target_avg_unc / avg_unc

    def set_grads(self, backbone=True, classifier=True, unc_module=False):
        # Freeze / unfreeeze parts of the model: backbone, classifier head and uncertainty head (if it exists)
        params_classifier = [p for p in self.model.get_classifier().parameters()]
        params_classifier_plus_backbone = [p for p in self.model.parameters()]
        params_classifier_plus_backbone_plus_unc = [p for p in self.parameters()]

        # There's no nice way to do diffs between those lists,
        # so we first set the unc parameters, then the subset of backbone, then the sub-subset of classifier
        for param in params_classifier_plus_backbone_plus_unc:
            param.requires_grad = unc_module
        for param in params_classifier_plus_backbone:
            param.requires_grad = backbone
        for param in params_classifier:
            param.requires_grad = classifier


class UncertaintyViaNorm(UncertaintyWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, *args, **kwargs):
        # In addition to whatever the model itself outputs (usually classes) (Tensor of shape [Batchsize, Classes])
        # also output an uncertainty estimate based on the norm of the embedding (Tensor of shape [Batchsize])
        features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
        # TODO: Some models have classifiers that use the unpooled embeddings (like ResNetv2)
        unc = 1 / features.norm(dim=-1)  # norms = certainty, so return their inverse
        unc = unc * self.unc_scaler
        out = self.model.get_classifier()(features)

        return out, unc, features

class UncertaintyViaJSD(UncertaintyWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, *args, **kwargs):
        if (isinstance(self.model, ResNetDropout) or isinstance(self.model, VisionTransformerDropout)) and not self.training:
            # Test-time dropout
            predictions = []
            for _ in range(self.model.num_dropout_samples):
                features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
                logits = self.model.get_classifier()(features)  # [B, C]
                predictions.append(logits.unsqueeze(0))

            # Stack predictions
            predictions = torch.cat(predictions, dim=0)  # [S, B, C]

            unc, out = get_unc_out(predictions, self.unc_scaler)
        elif isinstance(self.model, ShallowEnsembleWrapper):
            features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
            predictions = self.model.get_classifier()(features)  # [S, B, C]

            unc, out = get_unc_out(predictions, self.unc_scaler)
        else:
            if not (isinstance(self.model, ResNetDropout) or isinstance(self.model, VisionTransformerDropout)):
                raise ValueError(
                    f"Model has type {type(self.model)} but expected `ResNetDropout`"
                    ", ShallowEnsembleWrapper, or VisionTransformerDropout."
                )
            # In addition to whatever the model itself outputs (usually classes) (Tensor of shape [Batchsize, Classes])
            # also output an uncertainty estimate based on the JSD of the class distribution (Tensor of shape [Batchsize])
            features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
            out = self.model.get_classifier()(features)  # [B, C]
            class_prob = out.softmax(dim=-1)

            # Only calculate the entropy during training as a substitute uncertainty
            # value for dropout nets
            unc = entropy(class_prob)  # [B]
            unc = unc * self.unc_scaler

        return out, unc, features

class UncertaintyViaEntropy(UncertaintyWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, *args, **kwargs):
        if (isinstance(self.model, ResNetDropout) or isinstance(self.model, VisionTransformerDropout)) and not self.training:
            # Test-time dropout
            predictions = []
            for _ in range(self.model.num_dropout_samples):
                features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
                logits = self.model.get_classifier()(features)  # [B, C]
                predictions.append(logits.unsqueeze(0))

            # Stack predictions
            predictions = torch.cat(predictions, dim=0)  # [S, B, C]

            # Apply averaging
            out = F.softmax(predictions, dim=-1).mean(dim=0).log()
        elif isinstance(self.model, ShallowEnsembleWrapper):
            features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
            predictions = self.model.get_classifier()(features)  # [S, B, C]
            probs = F.softmax(predictions, dim=-1)  # [S, B, C]
            mean_probs = probs.mean(dim=0)  # [B, C]
            out = mean_probs.log()
        else:
            # In addition to whatever the model itself outputs (usually classes) (Tensor of shape [Batchsize, Classes])
            # also output an uncertainty estimate based on the entropy of the class distribution (Tensor of shape [Batchsize])
            features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
            out = self.model.get_classifier()(features)
        class_prob = out.softmax(dim=-1)

        entr = entropy(class_prob)
        entr = entr * self.unc_scaler

        return out, entr, features


class UncertaintyViaConst(UncertaintyWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, *args, **kwargs):
        # In addition to whatever the model itself outputs (usually classes) (Tensor of shape [Batchsize, Classes])
        # also output a constant uncertainty estimate (acting as baseline) (Tensor of shape [Batchsize])
        features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
        out = self.model.get_classifier()(features)
        unc = torch.ones(out.shape[0], device=out.device)
        unc = unc * self.unc_scaler

        return out, unc, features


class UncertaintyViaNetwork(UncertaintyWrapper):
    def __init__(self, model, stopgrad=False, *args, **kwargs):
        super().__init__(model)
        self.unc_module = UncertaintyNetwork(
            in_channels=model.num_features, *args, **kwargs
        )
        self.stopgrad=stopgrad

    def forward(self, *args, **kwargs):
        features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
        out = self.model.get_classifier()(features)
        unc = self.unc_module(features if not self.stopgrad else features.detach()).squeeze()
        unc = unc * self.unc_scaler

        return out, unc, features


class UncertaintyViaDeepNet(UncertaintyWrapper):
    def __init__(self, model, hook_layer_idxes=None, stopgrad=False, width=256, depth=4, num_hooks=4, *args, **kwargs):
        # model - a timm model
        # hook_layer_idxes - list of int, which depths to attach the hooks to
        # stopgrad - Boolean, whether to stop the gradient flowing back from the uncertainty network
        # width - int, width of each uncertainty network
        # num_hooks - int, number of hooks to attach uniformly to the network (if hook_layers is none)
        super().__init__(model)
        self.stopgrad = stopgrad

        if "resnet" in model.default_cfg["architecture"]:
            self.is_resnet = True
        elif "vit" in model.default_cfg["architecture"]:
            self.is_resnet = False
        else:
            raise NotImplementedError("Deep prednets only implemented for resnet and vit architectures. Use --unc_module='pred-net' instead.")

        # Register hooks to extract intermediate features
        self.feature_buffer = {}
        self.hook_layers = []
        self.layer_candidates = extract_layer_candidates(self.model, layertype=nn.ReLU if self.is_resnet else nn.LayerNorm)
        self.attach_hooks(hook_layer_idxes if hook_layer_idxes is not None else self.select_random_layers(num_hooks))

        # Initialize uncertainty network(s)
        self.feature_unc_modules = nn.ModuleDict({})
        self.feature_to_module = {}
        self.add_feature_unc_modules(width=width, stopgrad=self.stopgrad)
        self.unc_module = UncertaintyNetwork(in_channels=width*len(self.hook_layers), width=width, depth=depth-1, *args, **kwargs)

    def forward(self, *args, **kwargs):
        self.feature_buffer = {}
        ff = self.model_forward_features(*args, **kwargs)
        features = self.model_forward_head(ff, pre_logits=True)
        out = self.model.get_classifier()(features)
        unc_features = torch.cat(
            [self.feature_unc_modules[self.feature_to_module[hl[0]]](
                self.feature_buffer[hl[0]]
            ) for hl in self.hook_layers
            ], dim=1
        )
        # hl[0] is the name of the module, so iterating over the list (not dict!) self.hook_layers
        # ensures that we always concat in the same order
        unc = self.unc_module(unc_features).squeeze()
        unc = unc * self.unc_scaler

        return out, unc, features

    def select_random_layers(self, num_hooks=4, start_depth=0.1, end_depth=1.0):
        chosen_idx = np.linspace(start_depth * len(self.layer_candidates),
                                 min(end_depth * len(self.layer_candidates), len(self.layer_candidates) -1),
                                 num_hooks, dtype=int)
        return chosen_idx

    def attach_hooks(self, chosen_idx):
        def get_features(name):
            def hook(model, input, output):
                self.feature_buffer[name] = output

            return hook

        # Attach hooks to the chosen layers
        self.hook_layers = [self.layer_candidates[i] for i in chosen_idx]
        for layer in self.hook_layers:
            layer[1].register_forward_hook(get_features(layer[0]))

    def add_feature_unc_modules(self, width, stopgrad=False):
        # Get the feature map sizes
        empty_image = torch.zeros([1, *self.model.default_cfg["input_size"]],
                                  device=next(self.model.parameters()).device)
        with torch.no_grad():
            self.feature_buffer = {}
            self.model(empty_image)
            feature_sizes = {key: feature.shape[1] if self.is_resnet else feature.shape[-1] for key, feature in self.feature_buffer.items()}

        modules = {}
        self.feature_to_module = {}
        for i, (key, size) in enumerate(feature_sizes.items()):
            module_name = f"unc{i}"
            modules[module_name] = FeatureUncertaintyNetwork(size, width, pool=self.model.global_pool if self.is_resnet else VITAveragePool(),
                                                             use_layer_norm=False, stopgrad=stopgrad)
            self.feature_to_module[key] = module_name
        self.feature_unc_modules = nn.ModuleDict(modules)



def extract_layer_candidates(model, layertype=nn.ReLU):
    candidate_layers = []

    for name, module in model.named_modules():
        if isinstance(module, layertype):
            candidate_layers.append((name, module))

    return candidate_layers

class UncertaintyViaVAE(UncertaintyWrapper):
    def __init__(self, model, stopgrad=False, num_features=512, *args, **kwargs):
        super().__init__(model)
        self.unc_module = UncertaintyNetwork(
            in_channels=model.num_features, *args, **kwargs
        )
        self.embed_module = EmbedNetwork(
            in_channels=model.num_features, width=num_features
        )
        self.num_features = num_features
        self.model.num_features = num_features
        self.model.reset_classifier(self.model.num_classes)
        self.stopgrad=stopgrad

    def forward(self, *args, **kwargs):
        features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
        embeds = self.embed_module(features)
        out = self.model.get_classifier()(embeds)
        unc = self.unc_module(features if not self.stopgrad else features.detach()).squeeze()
        unc = unc * self.unc_scaler

        return out, unc, embeds

class UncertaintyViaHETXLCov(UncertaintyWrapper):  # TODO: initialize avg uncertainty?
    def __init__(self, model):
        super().__init__(model)

    def forward(self, *args, **kwargs):
        features = self.model_forward_head(self.model_forward_features(*args, **kwargs), pre_logits=True)
        out, unc = self.model.get_classifier()(features, calc_cov_log_det=True)
        unc = unc * self.unc_scaler

        return out, unc, features


class UncertaintyNetwork(nn.Module):
    def __init__(self, in_channels=2048, width=512, depth=3, init_prednet_zero=False) -> None:
        super().__init__()
        layers = [nn.Linear(in_channels, width),
            nn.LeakyReLU()]
        for i in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.LeakyReLU()
            ])
        layers.extend([
            nn.Linear(width, 1),
            nn.Softplus()
        ])
        self.unc_module = nn.Sequential(*layers)
        self.EPS = 1e-6

        if init_prednet_zero:
            self.unc_module.apply(self.init_weights_zero)

    def forward(self, input):
        return self.EPS + self.unc_module(input)

    def init_weights_zero(model, layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class EmbedNetwork(nn.Module):
    def __init__(self, in_channels=2048, width=512) -> None:
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(in_channels, width),
            nn.LeakyReLU(),
            nn.Linear(width, width),
            nn.LeakyReLU(),
            nn.Linear(width, width)
        )

    def forward(self, input):
        return self.embedder(input)


class VITAveragePool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.mean(dim=-2)


class FeatureUncertaintyNetwork(nn.Module):
    def __init__(self, in_size, width, pool, use_layer_norm=False, stopgrad=False):
        super().__init__()
        self.stopgrad = stopgrad
        self.pool = nn.Sequential(
            nn.LayerNorm(in_size) if use_layer_norm else nn.Identity(),
            pool,
            nn.LayerNorm(in_size) if use_layer_norm else nn.Identity()
        )
        self.net = nn.Sequential(
            nn.Linear(in_size, width),
            nn.LeakyReLU()
        )

    def forward(self, input):
        out = self.pool(input)
        if self.stopgrad:
            out = out.detach()
        out = self.net(out)
        return out


def entropy(probs):
    log_probs = probs.log()
    min_real = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs * probs
    
    return -p_log_p.sum(dim=-1)

def get_unc_out(predictions, unc_scaler):
    probs = F.softmax(predictions, dim=-1)  # [S, B, C]
    mean_probs = probs.mean(dim=0)  # [B, C]
    entropy_of_mean = entropy(mean_probs)  # [B]
    mean_of_entropy = entropy(probs).mean(dim=0)  # [B]

    unc = entropy_of_mean - mean_of_entropy
    unc = unc * unc_scaler

    # Apply averaging
    out = mean_probs.log()

    return unc, out