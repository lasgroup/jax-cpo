from typing import Union

import jax.numpy as jnp
import jmp


def get_mixed_precision_policy(precision):
  policy = ('params=float32,compute=float' + str(precision) + ',output=float' +
            str(precision))
  return jmp.get_policy(policy)


def discounted_cumsum(x: jnp.ndarray, discount: float) -> jnp.ndarray:
  """
  Compute a discounted cummulative sum of vector x. [x0, x1, x2] ->
  [x0 + discount * x1 + discount^2 * x2,
  x1 + discount * x2,
  x2]
  """
  # Divide by discount to have the first discount value from 1: [1, discount,
  # discount^2 ...]
  scales = jnp.cumprod(jnp.ones_like(x) * discount) / discount
  # Flip scales since jnp.convolve flips it as default.
  return jnp.convolve(x, scales[::-1])[-x.shape[0]:]


def inv_softplus(x: Union[jnp.ndarray, float]):
  return jnp.log(jnp.exp(x) - 1.0)
