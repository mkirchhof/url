import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
        self,
        input_shape,
        units,
        custom_random_features_initializer,
        num_inducing=1024,
        gp_kernel_scale=1.0,
        gp_output_bias=0.0,
        normalize_input=True,
        **gp_output_kwargs,
    ):
        super().__init__()
        self.units = units
        self.num_inducing = num_inducing

        self.normalize_input = normalize_input
        self.gp_input_scale = (
            1 / gp_kernel_scale**0.5 if gp_kernel_scale is not None else None
        )
        self.gp_feature_scale = (2 / num_inducing) ** 0.5

        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias

        self.custom_random_features_initializer = custom_random_features_initializer

        self.l2_regularization = 0
        self.gp_output_kwargs = gp_output_kwargs

        # self.custom_random_features_activation = None
        self.gp_cov_momentum = 0.999
        self.gp_cov_ridge_penalty = 1e-6

        # Default to Gaussian RBF kernel with orthogonal random features.
        self.random_features_bias_initializer = partial(
            nn.init.uniform_, a=0, b=2 * torch.pi
        )

        self.custom_random_features_activation = torch.cos

        # build

        self.bias_layer = nn.Parameter
        self.dense_layer = nn.Linear
        self.covariance_layer = LaplaceRandomFeatureCovariance
        self.input_normalization_layer = nn.LayerNorm

        if self.normalize_input:
            self._input_norm_layer = self.input_normalization_layer(input_shape)

        self._random_feature = self._make_random_feature_layer(input_shape)

        self._gp_cov_layer = self.covariance_layer(
            gp_feature_dim=self.num_inducing,
            momentum=self.gp_cov_momentum,
            ridge_penalty=self.gp_cov_ridge_penalty,
        )

        self._gp_output_layer = self.dense_layer(
            in_features=self.num_inducing, out_features=self.units, bias=False
        )
        output_initializer = self.gp_output_kwargs["kernel_initializer"]
        output_initializer(self._gp_output_layer.weight)

        self._gp_output_bias = self.bias_layer(
            torch.tensor([self.gp_output_bias] * self.units), requires_grad=False
        )

    def _make_random_feature_layer(self, input_shape):
        """Defines random feature layer depending on kernel type."""
        # Use user-supplied configurations.
        custom_random_feature_layer = self.dense_layer(
            in_features=input_shape,
            out_features=self.num_inducing,
        )
        self.custom_random_features_initializer(custom_random_feature_layer.weight)
        self.random_features_bias_initializer(custom_random_feature_layer.bias)
        custom_random_feature_layer.weight.requires_grad_(False)
        custom_random_feature_layer.bias.requires_grad_(False)

        return custom_random_feature_layer

    def reset_covariance_matrix(self):
        """Resets covariance matrix of the GP layer.

        This function is useful for reseting the model's covariance matrix at the
        begining of a new epoch.
        """
        self._gp_cov_layer.reset_precision_matrix()

    @staticmethod
    def mean_field_logits(logits, covmat, mean_field_factor):
        # Compute standard deviation.
        variances = covmat.diag()

        # Compute scaling coefficient for mean-field approximation.
        logits_scale = (1 + variances * mean_field_factor).sqrt()

        # Cast logits_scale to compatible dimension.
        logits_scale = logits_scale.reshape(-1, 1)

        return logits / logits_scale

    def forward(self, inputs):
        # Computes random features.
        gp_inputs = inputs
        if self.normalize_input:
            gp_inputs = self._input_norm_layer(gp_inputs)
        elif self.gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly
            # rescaling the input.
            gp_inputs = gp_inputs * self.gp_input_scale

        gp_feature = self._random_feature(gp_inputs).cos()

        # Computes posterior center (i.e., MAP estimate) and variance
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if self.training:
            return gp_output

        # Mean field
        with torch.no_grad():
            gp_covmat = self._gp_cov_layer(gp_feature)
        logits = self.mean_field_logits(gp_output, gp_covmat, mean_field_factor=1)
        return logits


class LaplaceRandomFeatureCovariance(nn.Module):
    def __init__(
        self,
        gp_feature_dim,
        momentum=0.999,
        ridge_penalty=1e-6,
    ):
        super().__init__()
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        # build

        # Posterior precision matrix for the GP's random feature coefficients
        self.precision_matrix = nn.Parameter(
            torch.zeros((gp_feature_dim, gp_feature_dim)), requires_grad=False
        )

        self.covariance_matrix = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty((gp_feature_dim, gp_feature_dim))),
            requires_grad=False,
        )

        # Boolean flag to indicate whether to update the covariance matrix (i.e.,
        # by inverting the newly updated precision matrix) during inference.
        self.if_update_covariance = False

    def update_feature_precision_matrix(self, gp_feature):
        """Computes the update precision matrix of feature weights."""
        batch_size = gp_feature.shape[0]

        # Computes batch-specific normalized precision matrix.
        precision_matrix_minibatch = gp_feature.T @ gp_feature

        # Updates the population-wise precision matrix.
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.momentum * self.precision_matrix
                + (1 - self.momentum) * precision_matrix_minibatch
            )
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch

        return precision_matrix_new

    def reset_precision_matrix(self):
        """Resets precision matrix to its initial value.

        This function is useful for reseting the model's covariance matrix at the
        begining of a new epoch.
        """
        self.precision_matrix.zero_()

    def update_feature_covariance_matrix(self):
        """Computes the feature covariance if self.if_update_covariance=True.

        GP layer computes the covariancce matrix of the random feature coefficient
        by inverting the precision matrix. Since this inversion op is expensive,
        we will invoke it only when there is new update to the precision matrix
        (where self.if_update_covariance will be flipped to `True`.).

        Returns:
        The updated covariance_matrix.
        """
        precision_matrix = self.precision_matrix
        covariance_matrix = self.covariance_matrix
        gp_feature_dim = precision_matrix.shape[0]

        # Compute covariance matrix update only when `if_update_covariance = True`.
        if self.if_update_covariance:
            covariance_matrix_updated = torch.linalg.inv(
                self.ridge_penalty
                * torch.eye(gp_feature_dim, device=precision_matrix.device)
                + precision_matrix
            )
        else:
            covariance_matrix_updated = covariance_matrix

        return covariance_matrix_updated

    def compute_predictive_covariance(self, gp_feature):
        """Computes posterior predictive variance.

        Approximates the Gaussian process posterior using random features.
        Given training random feature Phi_tr (num_train, num_hidden) and testing
        random feature Phi_ts (batch_size, num_hidden). The predictive covariance
        matrix is computed as (assuming Gaussian likelihood):

        s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

        where s is the ridge factor to be used for stablizing the inverse, and I is
        the identity matrix with shape (num_hidden, num_hidden).

        Args:
        gp_feature: (torch.Tensor) The random feature of testing data to be used for
            computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
        (torch.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
        """
        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = (
            self.covariance_matrix @ gp_feature.T
        ) * self.ridge_penalty
        gp_cov_matrix = gp_feature @ cov_feature_product

        return gp_cov_matrix

    def forward(self, inputs):
        """Minibatch updates the GP's posterior precision matrix estimate.

        Args:
        inputs: (torchf.Tensor) GP random features, shape (batch_size,
            gp_hidden_size).
        logits: (torch.Tensor) Pre-activation output from the model. Needed
            for Laplace approximation under a non-Gaussian likelihood.
        training: (torch.bool) whether or not the layer is in training mode. If in
            training mode, the gp_weight covariance is updated using gp_feature.

        Returns:
        gp_stddev (tf.Tensor): GP posterior predictive variance,
            shape (batch_size, batch_size).
        """
        batch_size = inputs.shape[0]

        if self.training:
            # Computes the updated feature precision matrix.
            precision_matrix_updated = self.update_feature_precision_matrix(
                gp_feature=inputs
            )

            # Updates precision matrix.
            self.precision_matrix.data = precision_matrix_updated

            # Enables covariance update in the next inference call.
            self.if_update_covariance = True

            # Return null estimate during training.
            return torch.eye(batch_size, device=inputs.device)
        else:
            # Lazily computes feature covariance matrix during inference.
            covariance_matrix_updated = self.update_feature_covariance_matrix()

            # Store updated covariance matrix.
            self.covariance_matrix.data = covariance_matrix_updated

            # Disable covariance update in future inference calls (to avoid the
            # expensive torch.linalg.inv op) unless there are new update to precision
            # matrix.
            self.if_update_covariance = False

            return self.compute_predictive_covariance(gp_feature=inputs)


class MonteCarloDropout(nn.Module):
    """Defines the Monte Carlo dropout layer."""

    def __init__(self, dropout_rate, use_mc_dropout, filterwise_dropout):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_mc_dropout = use_mc_dropout
        self.filterwise_dropout = filterwise_dropout

    def forward(self, inputs):
        if self.use_mc_dropout or self.training:
            if self.filterwise_dropout:
                noise_shape = [inputs.shape[0], 1, 1, inputs.shape[3]]
            else:
                noise_shape = inputs.shape
            prob_tensor = torch.ones((noise_shape), device=inputs.device)
            bernoulli_mask = torch.bernoulli(prob_tensor).to(inputs.device)

            return bernoulli_mask * inputs
        else:
            return inputs


class Conv2dNormedWrapper(nn.Module):
    """Implements spectral normalization for Conv2d layer."""

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
    ):
        super().__init__()
        self.iteration = iteration
        self.norm_multiplier = norm_multiplier
        self.layer = layer

        self.built = False

    def build(self, input_shape):
        # build
        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel)
        weight = self.layer.weight.data
        weight_shape = [
            weight.shape[2],
            weight.shape[3],
            weight.shape[1],
            weight.shape[0],
        ]
        self.stride = self.layer.stride

        # Set the dimensions of u and v vectors
        uv_dim = 1

        # Resolve shapes.
        in_height = input_shape[2]
        in_width = input_shape[3]
        in_channel = weight_shape[2]

        out_height = in_height // self.stride[0]
        out_width = in_width // self.stride[1]
        out_channel = weight_shape[3]

        in_shape = (uv_dim, in_channel, in_height, in_width)
        out_shape = (uv_dim, out_channel, out_height, out_width)

        self.v = nn.Parameter(torch.randn(in_shape, device=self.layer.weight.device), requires_grad=False)
        self.u = nn.Parameter(torch.randn(out_shape, device=self.layer.weight.device), requires_grad=False)

        self.built = True

    def forward(self, inputs):
        with torch.no_grad():
            if not self.built:
                self.build(inputs.shape)

            self.update_weights()
        output = self.layer(inputs)
        with torch.no_grad():
            self.restore_weights()

        return output

    @staticmethod
    def calc_same_padding(input_shape, filter_shape, stride):
        """Calculates padding values for 'SAME' padding for conv2d.
        
        Args:
            input_shape (tuple or list): Shape of the input data. [batch, channels, height, width]
            filter_shape (tuple or list): Shape of the filter/kernel. [out_channels, in_channels, kernel_height, kernel_width]
            stride (int or tuple): Stride of the convolution operation.

        Returns:
            padding (tuple): Tuple representing padding (padding_left, padding_right, padding_top, padding_bottom)
        """
        if isinstance(stride, int):
            stride_height = stride_width = stride
        else:
            stride_height, stride_width = stride

        in_height, in_width = input_shape[2], input_shape[3]
        filter_height, filter_width = filter_shape[2], filter_shape[3]

        if in_height % stride_height == 0:
            pad_along_height = max(filter_height - stride_height, 0)
        else:
            pad_along_height = max(filter_height - (in_height % stride_height), 0)
            
        if in_width % stride_width == 0:
            pad_along_width = max(filter_width - stride_width, 0)
        else:
            pad_along_width = max(filter_width - (in_width % stride_width), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    def update_weights(self):
        """Computes power iteration for convolutional filters."""
        self.saved_weight = self.layer.weight.data

        # Initialize u, v vectors.
        u_hat = self.u.data
        v_hat = self.v.data

        padding = self.calc_same_padding(
            u_hat.shape, self.saved_weight.shape, self.stride
        )
        agg_padding = ((padding[0] + padding[1]) // 2, (padding[2] + padding[3]) // 2)

        if self.training:
            for _ in range(self.iteration):
                # Updates v.
                output_padding = (int(self.stride[0] == 2), int(self.stride[1] == 2))
                v_ = F.conv_transpose2d(
                    input=u_hat,
                    weight=self.saved_weight,
                    stride=self.stride,
                    padding=agg_padding,
                    output_padding=output_padding,
                )

                v_hat = F.normalize(v_.reshape(1, -1))
                v_hat = v_hat.reshape(v_.shape)

                # Updates u.
                padded_v_hat = F.pad(v_hat, padding)
                u_ = F.conv2d(
                    input=padded_v_hat,
                    weight=self.saved_weight,
                    stride=self.stride,
                )

                u_hat = F.normalize(u_.reshape(1, -1))
                u_hat = u_hat.reshape(u_.shape)

        padded_v_hat = F.pad(v_hat, padding)
        v_w_hat = F.conv2d(
            input=padded_v_hat, weight=self.saved_weight, stride=self.stride
        )

        sigma = v_w_hat.flatten() @ u_hat.flatten()

        if self.norm_multiplier / sigma < 1:
            weight_norm = self.norm_multiplier / sigma * self.saved_weight
        else:
            weight_norm = self.saved_weight

        self.u.data = u_hat
        self.v.data = v_hat
        self.layer.weight.data = weight_norm

    def restore_weights(self):
        """Restores layer weights to maintain gradient update."""
        self.layer.weight.data = self.saved_weight


def make_conv2d_layer(use_spec_norm, spec_norm_iteration, spec_norm_bound):
    """Defines type of Conv2D layer to use based on spectral normalization."""
    if not use_spec_norm:
        return nn.Conv2d
    else:

        def conv_2d_normed(*args, **kwargs):
            return Conv2dNormedWrapper(
                nn.Conv2d(*args, **kwargs),
                iteration=spec_norm_iteration,
                norm_multiplier=spec_norm_bound,
            )

        return conv_2d_normed
