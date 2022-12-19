from typing import Callable, Tuple, Any, Dict, NamedTuple, TypeVar
import functools

import chex
import haiku as hk
import jax.numpy as jnp
import optax

PRNGKey = jnp.ndarray

from typing import Union

import jax
import jax.numpy as jnp
import jmp

T = TypeVar("T")


class LearningState(NamedTuple):
  params: Union[hk.Params, chex.ArrayTree]
  opt_state: optax.OptState


class Learner:

  def __init__(self, model: Union[hk.Transformed, hk.MultiTransformed,
                                  chex.ArrayTree], seed: PRNGKey,
               optimizer_config: Dict, precision: jmp.Policy,
               *input_example: Any):
    self.optimizer = optax.flatten(
        optax.chain(
            optax.clip_by_global_norm(
                optimizer_config.get('clip', float('inf'))),
            optax.scale_by_adam(eps=optimizer_config.get('eps', 1e-8)),
            optax.scale(-optimizer_config.get('lr', 1e-3))))
    self.model = model
    if isinstance(model, (hk.Transformed, hk.MultiTransformed)):
      self.params = self.model.init(seed, *input_example)
    else:
      self.params = model
    self.opt_state = self.optimizer.init(self.params)
    self.precision = precision

  @property
  def apply(self) -> Union[Callable, Tuple[Callable]]:
    if isinstance(self.model, (hk.Transformed, hk.MultiTransformed)):
      return self.model.apply
    else:
      return lambda: self.model

  @property
  def state(self):
    return LearningState(self.params, self.opt_state)

  @state.setter
  def state(self, state: LearningState):
    self.params = state.params
    self.opt_state = state.opt_state

  def grad_step(self, grads, state: LearningState):
    params, opt_state = state
    grads = self.precision.cast_to_param(grads)
    updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    grads_finite = jmp.all_finite(grads)
    new_params, new_opt_state = select_tree(grads_finite,
                                                (new_params, new_opt_state),
                                                (params, opt_state))
    return LearningState(new_params, new_opt_state)


def select_tree(pred: jnp.ndarray, a: T, b: T) -> T:
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_map(functools.partial(jax.lax.select, pred), a, b)

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
