from typing import Sequence, Optional, Callable, Union

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from jax_cpo import nets

tfd = tfp.distributions
tfb = tfp.bijectors


class Actor(hk.Module):
    def __init__(
        self,
        output_size: Sequence[int],
        layers: Sequence[int],
        min_stddev: float,
        max_stddev: float,
        initialization: str = "glorot",
        activation: Union[str, Callable[[jnp.ndarray], jnp.ndarray]] = jnn.relu,
    ):
        super().__init__()
        self.output_size = output_size
        self.layers = layers
        self._min_stddev = min_stddev
        self._max_stddev = max_stddev
        self._initialization = initialization
        self._activation = activation if callable(activation) else eval(activation)

    def __call__(self, observation: jnp.ndarray):
        x = nets.mlp(
            observation,
            output_sizes=tuple(self.layers) + tuple(self.output_size),
            initializer=nets.initializer(self._initialization),
            activation=self._activation,
        )
        mu, stddev = (
            x,
            hk.get_parameter(
                "pi_stddev", (x.shape[-1],), x.dtype, hk.initializers.Constant(-0.5)
            ),
        )
        if stddev.ndim == 1:
            stddev = jnp.expand_dims(stddev, 0)
        stddev = jnp.exp(stddev)
        multivariate_normal_diag = tfd.MultivariateNormalDiag(mu, stddev)
        return multivariate_normal_diag


class DenseDecoder(hk.Module):
    def __init__(
        self,
        output_size: Sequence[int],
        layers: Sequence[int],
        dist: str,
        initialization: str = "glorot",
        activation: Union[str, Callable[[jnp.ndarray], jnp.ndarray]] = jnn.relu,
        name: Optional[str] = None,
    ):
        super(DenseDecoder, self).__init__(name)
        self.output_size = output_size
        self.layers = layers
        self._dist = dist
        self._initialization = initialization
        self._activation = activation if callable(activation) else eval(activation)

    def __call__(self, x: jnp.ndarray):
        x = nets.mlp(
            x,
            output_sizes=tuple(self.layers) + tuple(self.output_size),
            initializer=nets.initializer(self._initialization),
            activation=self._activation,
        )
        x = jnp.squeeze(x, axis=-1)
        dist = dict(
            normal=lambda mu: tfd.Normal(mu, 1.0), bernoulli=lambda p: tfd.Bernoulli(p)
        )[self._dist]
        return tfd.Independent(dist(x), 0)
