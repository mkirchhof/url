import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from timm.layers import set_layer_config
from ._pretrained import PretrainedCfg, split_model_name_tag
from ._helpers import load_checkpoint
from ._hub import load_model_config_from_hf
from ._registry import is_model, model_entrypoint
from ._uncertainizer import (
    UncertaintyViaNorm,
    UncertaintyViaNetwork,
    UncertaintyViaEntropy,
    UncertaintyViaConst,
    ShallowEnsembleWrapper,
    UncertaintyViaHETXLCov,
    UncertaintyViaJSD,
    UncertaintyViaVAE,
    UncertaintyViaDeepNet
)


__all__ = ['parse_model_name', 'safe_model_name', 'create_model']


def parse_model_name(model_name):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def safe_model_name(model_name, remove_source=True):
    # return a filename / path safe model name
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def add_unc_module(model, unc_module, unc_width, unc_depth=3, init_prednet_zero=False, stopgrad=False):
    if unc_module == "embed-norm":
        return UncertaintyViaNorm(model)
    elif unc_module == "pred-net" or unc_module == "prednet":
        return UncertaintyViaNetwork(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module == "deep-prednet":
        return UncertaintyViaDeepNet(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module.startswith("pred-net-layer_") or unc_module.startswith("prednet-layer_"):
        idxes = [int(unc_module.split("_")[-1])]
        return UncertaintyViaDeepNet(model, hook_layer_idxes=idxes, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module == "vae":
        return UncertaintyViaVAE(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module == "class-entropy":
        return UncertaintyViaEntropy(model)
    elif unc_module == "jsd":
        return UncertaintyViaJSD(model)
    elif unc_module == "hetxl-det":
        return UncertaintyViaHETXLCov(model)
    elif unc_module == "none":
        return UncertaintyViaConst(model)
    else:
        raise NotImplementedError(f"Argument --unc_module {unc_module} is not implemented.")


def create_model(
        model_name: str,
        unc_module:str = "none",
        unc_width:int = 512,
        unc_depth:int = 3,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        num_heads: int = 1,
        init_prednet_zero=False,
        stopgrad=False,
        **kwargs,
):
    """Create a model

    Lookup model's entrypoint function and pass relevant args to create a new model.

    **kwargs will be passed through entrypoint fn to timm.models.build_model_with_cfg()
    and then the model class __init__(). kwargs values set to None are pruned before passing.

    Args:
        model_name (str): name of model to instantiate
        unc_module (str): type of the uncertainty estimator
        unc_width (int): Width of the uncertainty estimation network (if used)
        unc_depth (int): Number of hidden layers in uncertainty estimation network (if used)
        pretrained (bool): load pretrained ImageNet-1k weights if true
        pretrained_cfg (Union[str, dict, PretrainedCfg]): pass in external pretrained_cfg for model
        pretrained_cfg_overlay (dict): replace key-values in base pretrained_cfg with these
        checkpoint_path (str): path of checkpoint to load _after_ the model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are consumed by builder or model __init__()
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == 'hf-hub':
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    if not "hetxl" in model_name:
        kwargs.pop("rank_V", None)
        kwargs.pop("c_mult", None)
    
    if not "sngp" in model_name:
        kwargs.pop("gp_input_normalization", None)
        kwargs.pop("gp_cov_discount_factor", None)
        kwargs.pop("use_spec_norm", None)
        kwargs.pop("spec_norm_bound", None)
    
    if "sngp" in model_name and not "resnet" in model_name:
        if kwargs.pop("use_spec_norm", None):
            raise ValueError("Spectral normalization is not implemented for ViTs.")
        kwargs.pop("spec_norm_bound", None)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if num_heads > 1:
        model = ShallowEnsembleWrapper(model, num_heads=num_heads)

    model = add_unc_module(model, unc_module, unc_width, unc_depth, init_prednet_zero, stopgrad)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
