import functools
from typing import Any, Callable, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import jmp


T = TypeVar("T")


class LearningState(NamedTuple):
    params: hk.Params | chex.ArrayTree
    opt_state: optax.OptState


class Learner:
    def __init__(
        self,
        model: hk.Transformed | hk.MultiTransformed | chex.ArrayTree,
        seed: jax.Array,
        optimizer_config: dict,
        *input_example: Any,
    ):
        self.optimizer = optax.flatten(
            optax.chain(
                optax.clip_by_global_norm(optimizer_config.get("clip", float("inf"))),
                optax.scale_by_adam(eps=optimizer_config.get("eps", 1e-8)),
                optax.scale(-optimizer_config.get("lr", 1e-3)),
            )
        )
        self.model = model
        if isinstance(model, (hk.Transformed, hk.MultiTransformed)):
            self.params = self.model.init(seed, *input_example)
        else:
            self.params = model
        self.opt_state = self.optimizer.init(self.params)

    @property
    def apply(self) -> Callable:
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
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        grads_finite = jmp.all_finite(grads)
        new_params, new_opt_state = select_tree(
            grads_finite, (new_params, new_opt_state), (params, opt_state)
        )
        return LearningState(new_params, new_opt_state)


def select_tree(pred: jnp.ndarray, a: T, b: T) -> T:
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_map(functools.partial(jax.lax.select, pred), a, b)


def all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = list(map(jnp.isfinite, leaves))
        leaves = list(map(jnp.all, leaves))
        return jnp.stack(list(leaves)).all()


def update_if(pred, update, fallback):
    return jax.tree_map(lambda x, y: jax.lax.select(pred, x, y), update, fallback)
