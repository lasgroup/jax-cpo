from typing import Callable, Tuple

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp


def initializer(name: str) -> hk.initializers.Initializer:
  return {
      'glorot': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
      'he': hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
  }[name]


def mlp(x: jnp.ndarray,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
        initializer: hk.initializers.Initializer = initializer('glorot'),
        output_sizes: Tuple = (128, 128, 1)):
  x = hk.nets.MLP(output_sizes, activation=activation, w_init=initializer)(x)
  return x
