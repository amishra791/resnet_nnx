import jax
from jax import numpy as jnp
from flax import nnx
from flax import linen as nn
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding


STAGE_SIZES = {34: [3, 4, 6, 3]}


class BatchNorm(nnx.Module):
    """Implementaiton of BatchNorm for ConvLayers only."""

    def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.9):
        normalization_shape = (1, 1, 1, n_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nnx.Param(
            jnp.ones(normalization_shape), sharding=(None, None, None, None)
        )
        self.beta = nnx.Param(
            jnp.zeros(normalization_shape), sharding=(None, None, None, None)
        )

        # Init the moving mean to be one and variance to be zero
        self.moving_mean = nnx.BatchStat(
            jnp.zeros(normalization_shape), sharding=(None, None, None, None)
        )
        self.moving_var = nnx.BatchStat(
            jnp.ones(normalization_shape), sharding=(None, None, None, None)
        )

        self.eps = eps
        self.momentum = momentum

        # To make the module compatible with `.eval()`, need to define these parameters.
        self.use_running_average = False

    def __call__(self, x_bhwc: jax.Array) -> jax.Array:
        if self.use_running_average:
            x_hat_bhwc = (x_bhwc - self.moving_mean) / jnp.sqrt(
                self.moving_var + self.eps
            )
        else:
            # calculate in-batch mean and variance and use those values to normalize input
            in_batch_mean = jnp.mean(x_bhwc, axis=(0, 1, 2), keepdims=True)
            in_batch_var = jnp.var(x_bhwc, axis=(0, 1, 2), keepdims=True)

            x_hat_bhwc = (x_bhwc - in_batch_mean) / jnp.sqrt(in_batch_var + self.eps)

            self.moving_mean = (
                self.momentum * self.moving_mean + (1 - self.momentum) * in_batch_mean
            )
            self.moving_var = (
                self.momentum * self.moving_var + (1 - self.momentum) * in_batch_var
            )

        return self.gamma * x_hat_bhwc + self.beta


class ResNetStem(nnx.Module):
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()

        self.conv = nnx.Conv(
            in_features,
            64,
            kernel_size=(7, 7),
            strides=2,
            padding=[(2, 3), (2, 3)],
            rngs=rngs,
            kernel_init=nnx.with_partitioning(init_fn, (None, None, None, None)),
        )

    def __call__(self, x_bhwc) -> jax.Array:
        x_bhwc = self.conv(x_bhwc)
        x_bhwc = nn.max_pool(
            x_bhwc, window_shape=(3, 3), strides=(2, 2), padding=[(0, 1), (0, 1)]
        )
        return x_bhwc


class ResNetBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()

        self.in_features = in_features
        self.out_features = out_features

        if self.in_features != self.out_features:
            assert self.in_features * 2 == self.out_features
            strides = (2, 2)
        else:
            strides = (1, 1)

        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            rngs=rngs,
            kernel_init=nnx.with_partitioning(init_fn, (None, None, None, None)),
        )
        self.bn1 = BatchNorm(out_features)
        self.conv2 = nnx.Conv(
            out_features,
            out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
            kernel_init=nnx.with_partitioning(init_fn, (None, None, None, None)),
        )
        self.bn2 = BatchNorm(out_features)

        if self.in_features != self.out_features:
            assert self.in_features * 2 == self.out_features
            self.conv_proj = nnx.Conv(
                in_features,
                out_features,
                kernel_size=(1, 1),
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
                kernel_init=nnx.with_partitioning(init_fn, (None, None, None, None)),
            )
            self.bn3 = BatchNorm(out_features)

    def __call__(self, x_bhwc: jax.Array) -> jax.Array:
        x_proj_bhwc = x_bhwc
        z_bhwc = self.conv1(x_bhwc)
        z_bhwc = self.bn1(z_bhwc)
        z_bhwc = nnx.relu(z_bhwc)
        z_bhwc = self.conv2(z_bhwc)
        z_bhwc = self.bn2(z_bhwc)

        if self.in_features != self.out_features:
            assert self.in_features * 2 == self.out_features
            x_proj_bhwc = self.conv_proj(x_proj_bhwc)
            x_proj_bhwc = self.bn3(x_proj_bhwc)

        return nnx.relu(z_bhwc + x_proj_bhwc)


class ResNetClassifier(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()

        self.dense = nnx.Linear(
            512,
            1000,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
        )

    def __call__(self, x_bhwc):
        x_bc = jnp.mean(x_bhwc, axis=(1, 2))
        x_bC = self.dense(x_bc)

        return x_bC


class ResNet(nnx.Module):
    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.stem = ResNetStem(in_features=3, rngs=rngs)
        self.module_blocks = []

        in_features = 64
        for stage_idx, stage_size in enumerate(STAGE_SIZES[num_layers]):
            for block_idx in range(stage_size):
                if stage_idx > 0 and block_idx == 0:
                    out_features = in_features * 2
                else:
                    out_features = in_features
                self.module_blocks.append(ResNetBlock(in_features, out_features, rngs))
                in_features = out_features

        self.classifier = ResNetClassifier(rngs=rngs)

    def __call__(self, x_bhwc: jax.Array) -> jax.Array:
        x_bhwc = self.stem(x_bhwc)
        for module in self.module_blocks:
            x_bhwc = module(x_bhwc)
        x_bhwc = self.classifier(x_bhwc)
        return x_bhwc


@nnx.jit
def create_sharded_model():
    model = ResNet(num_layers=34, rngs=nnx.Rngs(params=0))  # Unsharded at this moment.
    state = nnx.state(model)  # The model's state, a pure pytree.
    pspecs = nnx.get_partition_spec(state)  # Strip out the annotations from state.
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)  # The model is sharded now!
    return model
