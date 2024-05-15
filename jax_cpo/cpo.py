from functools import partial
from types import SimpleNamespace
from typing import Callable, NamedTuple

import chex
import optax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces
from jax.scipy import sparse
from tensorflow_probability.substrates import jax as tfp

from jax_cpo.rl.epoch_summary import EpochSummary
from jax_cpo.rl.metrics import MetricsMonitor
from jax_cpo.rl.trajectory import TrajectoryData
from jax_cpo import episodic_trajectory_buffer as etb
from jax_cpo.rl.learner import Learner, LearningState
from jax_cpo.rl.types import Report

tfd = tfp.distributions


@partial(jax.vmap, in_axes=[0, None])
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
    return jnp.convolve(x, scales[::-1])[-x.shape[0] :]


class Evaluation(NamedTuple):
    advantage: np.ndarray
    return_: np.ndarray
    cost_advantage: np.ndarray
    cost_return: np.ndarray


class CPO:
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        config: SimpleNamespace,
        actor: hk.Transformed,
        critic: hk.Transformed,
        safety_critic: hk.Transformed,
    ):
        self.config = config
        self.rng_seq = hk.PRNGSequence(config.training.seed)
        num_steps = (
            self.config.training.time_limit // self.config.training.action_repeat
        )
        self.buffer = etb.EpisodicTrajectoryBuffer(
            self.config.agent.num_trajectories,
            num_steps,
            observation_space.shape,
            action_space.shape,
        )
        self.actor = Learner(
            actor,
            next(self.rng_seq),
            config.agent.actor_opt,
            observation_space.sample(),
        )
        self.critic = Learner(
            critic,
            next(self.rng_seq),
            config.agent.critic_opt,
            observation_space.sample(),
        )
        self.safety_critic = Learner(
            safety_critic,
            next(self.rng_seq),
            config.agent.critic_opt,
            observation_space.sample(),
        )
        self.margin = 0.0
        self.metrics_monitor = MetricsMonitor()

    def __call__(
        self, observation: np.ndarray, train: bool, *args, **kwargs
    ) -> np.ndarray:
        if self.buffer.full and train:
            self.train(self.buffer.dump())
        action = self.policy(observation, self.actor.params, next(self.rng_seq), train)
        return action

    @partial(jax.jit, static_argnums=(0, 4))
    def policy(
        self,
        observation: jnp.ndarray,
        params: hk.Params,
        key: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        policy = self.actor.apply(params, observation)
        action = policy.sample(seed=key) if training else policy.mode()
        return action

    def observe(self, trajectory: TrajectoryData):
        self.buffer.add(trajectory)

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        return Report(metrics)

    def train(self, trajectory_data: TrajectoryData):
        observation = jnp.concatenate(
            (trajectory_data.observation, trajectory_data.next_observation[:, -1:]),
            axis=1,
        )
        eval_ = self.evaluate_trajectories(
            self.critic.params,
            self.safety_critic.params,
            observation,
            trajectory_data.reward,
            trajectory_data.cost,
        )
        constraint = trajectory_data.cost.sum(1).mean()
        # https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L260
        c = constraint - self.config.training.cost_limit
        self.margin = max(0, self.margin + self.config.agent.margin_lr * c)
        c += self.margin
        c /= self.config.training.time_limit + 1e-8
        self.actor.state, actor_report = self.update_actor(
            self.actor.state,
            trajectory_data.observation,
            trajectory_data.action,
            eval_.advantage,
            eval_.cost_advantage,
            c,
        )
        self.critic.state, critic_report = self.update_critic(
            self.critic.state, trajectory_data.observation, eval_.return_
        )
        if self.safe:
            self.safety_critic.state, safety_report = self.update_safety_critic(
                self.safety_critic.state,
                trajectory_data.observation,
                eval_.cost_return,
            )
            critic_report.update(safety_report)
        info = {**actor_report, **critic_report, "agent/margin": self.margin}
        for k, v in info.items():
            self.metrics_monitor[k] = np.asarray(v).mean()

    @partial(jax.jit, static_argnums=0)
    def update_actor(self, state: LearningState, *args) -> tuple[LearningState, dict]:
        observation, action, advantage, cost_advantage, c = args
        old_pi = self.actor.apply(state.params, observation)
        old_pi_logprob = old_pi.log_prob(action)
        g, b, old_pi_loss, old_surrogate_cost = self._cpo_grads(
            state.params, observation, action, advantage, cost_advantage, old_pi_logprob
        )
        p, unravel_tree = jax.flatten_util.ravel_pytree(state.params)

        def d_kl_hvp(x):
            d_kl = (
                lambda p: self.actor.apply(unravel_tree(p), observation)
                .kl_divergence(old_pi)
                .mean()
            )
            # Ravel the params so every computation from now on is made on actual
            # vectors.
            return hvp(d_kl, (p,), (x,))

        direction, optim_case = step_direction(
            g,
            b,
            c,
            d_kl_hvp,
            self.config.agent.target_kl,
            self.config.training.safe,
            self.config.agent.damping_coeff,
        )

        def evaluate_policy(params):
            new_pi_loss, new_surrogate_cost = self.policy_loss(
                params, observation, action, advantage, cost_advantage, old_pi_logprob
            )
            pi = self.actor.apply(params, observation)
            kl_d = pi.kl_divergence(old_pi).mean()
            return new_pi_loss, new_surrogate_cost, kl_d

        new_params, info = backtracking(
            direction,
            evaluate_policy,
            old_pi_loss,
            old_surrogate_cost,
            optim_case,
            c,
            state.params,
            self.config.training.safe,
            self.config.agent.backtrack_iters,
            self.config.agent.backtrack_coeff,
            self.config.agent.target_kl,
        )
        return LearningState(new_params, self.actor.opt_state), info

    def _cpo_grads(
        self,
        pi_params: hk.Params,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        advantage: jnp.ndarray,
        cost_advantage: jnp.ndarray,
        old_pi_logprob: jnp.ndarray,
    ):
        # Take gradients of the objective and surrogate cost w.r.t. pi_params.
        jac = jax.jacobian(self.policy_loss)(
            pi_params, observation, action, advantage, cost_advantage, old_pi_logprob
        )
        out = self.policy_loss(
            pi_params, observation, action, advantage, cost_advantage, old_pi_logprob
        )
        old_pi_loss, surrogate_cost_old = out
        g, b = jac
        return g, b, old_pi_loss, surrogate_cost_old

    def policy_loss(self, params: hk.Params, *args):
        observation, action, advantage, cost_advantage, old_pi_logprob = args
        pi = self.actor.apply(params, observation)
        logprob = pi.log_prob(action)
        ratio = jnp.exp(logprob - old_pi_logprob)
        surr_advantage = ratio * advantage
        objective = (
            surr_advantage + self.config.agent.entropy_regularization * pi.entropy()
        )
        surrogate_cost = ratio * cost_advantage
        return -objective.mean(), surrogate_cost.mean()

    @partial(jax.jit, static_argnums=0)
    def update_safety_critic(
        self,
        state: LearningState,
        observation: jnp.ndarray,
        cost_return: jnp.ndarray,
    ) -> tuple[LearningState, dict]:
        def safety_critic_loss(params: hk.Params) -> float:
            return (
                -self.safety_critic.apply(params, observation)
                .log_prob(cost_return)
                .mean()
            )

        def update(critic_state: LearningState):
            loss, grads = jax.value_and_grad(safety_critic_loss)(critic_state.params)
            new_critic_state = self.safety_critic.grad_step(grads, critic_state)
            return new_critic_state, {
                "agent/safety_critic/loss": loss,
                "agent/safety_critic/grad": optax.global_norm(grads),
            }

        return jax.lax.scan(
            lambda state, _: update(state),
            state,
            jnp.arange(self.config.agent.vf_iters),
        )

    @partial(jax.jit, static_argnums=0)
    def update_critic(
        self, state: LearningState, observation: jnp.ndarray, return_: jnp.ndarray
    ) -> tuple[LearningState, dict]:
        def critic_loss(params: hk.Params):
            return -self.critic.apply(params, observation).log_prob(return_).mean()

        def update(critic_state: LearningState):
            loss, grads = jax.value_and_grad(critic_loss)(critic_state.params)
            new_critic_state = self.critic.grad_step(grads, critic_state)
            return new_critic_state, {
                "agent/critic/loss": loss,
                "agent/critic/grad": optax.global_norm(grads),
            }

        return jax.lax.scan(
            lambda state, _: update(state),
            state,
            jnp.arange(self.config.agent.vf_iters),
        )

    @partial(jax.jit, static_argnums=0)
    def evaluate_trajectories(
        self,
        critic_params: hk.Params,
        safety_critic_params: hk.Params,
        observation: jnp.ndarray,
        reward: jnp.ndarray,
        cost: jnp.ndarray,
    ) -> Evaluation:
        value = self.critic.apply(critic_params, observation).mode()
        diff = reward + (self.config.agent.discount * value[..., 1:] - value[..., :-1])
        advantage = discounted_cumsum(
            diff, self.config.agent.lambda_ * self.config.agent.discount
        )
        mean, stddev = advantage.mean(), advantage.std()
        return_ = discounted_cumsum(reward, self.config.agent.discount)
        advantage = (advantage - mean) / (stddev + 1e-8)
        if not self.safe:
            return Evaluation(
                advantage, return_, jnp.zeros_like(advantage), jnp.zeros_like(return_)
            )
        cost_value = self.safety_critic.apply(safety_critic_params, observation).mode()
        cost_return = discounted_cumsum(cost, self.config.agent.cost_discount)
        diff = cost + (
            self.config.agent.cost_discount * cost_value[..., 1:] - cost_value[..., :-1]
        )
        cost_advantage = discounted_cumsum(
            diff, self.config.agent.lambda_ * self.config.agent.cost_discount
        )
        # Centering advantage, but not normalize, as in
        # https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/buffer.py#L71
        cost_advantage -= cost_advantage.mean()
        return Evaluation(advantage, return_, cost_advantage, cost_return)

    @property
    def safe(self):
        return self.config.training.safe


def step_direction(
    g: chex.ArrayTree,
    b: chex.ArrayTree,
    c: jnp.ndarray,
    d_kl_hvp: Callable,
    target_kl: float,
    safe: bool,
    damping_coeff: float = 0.0,
):
    # Implementation of CPO step direction is based on the implementation @
    # https://github.com/openai/safety-starter-agents
    # Take gradients of the objective and surrogate cost w.r.t. pi_params.
    g, unravel_tree = jax.flatten_util.ravel_pytree(g)
    b, _ = jax.flatten_util.ravel_pytree(b)
    # Add damping to hvp, as in TRPO.
    damped_d_kl_hvp = lambda v: d_kl_hvp(v) + damping_coeff * v
    v = sparse.linalg.cg(damped_d_kl_hvp, g, maxiter=10)[0]
    approx_g = damped_d_kl_hvp(v)
    q = jnp.dot(v, approx_g)

    def trpo():
        w, r, s, A, B = jnp.zeros_like(v), 0.0, 0.0, 0.0, 0.0
        optim_case = 4
        return optim_case, w, r, s, A, B

    def cpo():
        w = sparse.linalg.cg(damped_d_kl_hvp, b, maxiter=10)[0]
        r = jnp.dot(w, approx_g)
        s = jnp.dot(w, damped_d_kl_hvp(w))
        A = q - r**2 / s
        B = 2.0 * target_kl - c**2 / s
        # Implementation of the elif conditions in https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L282
        optim_case = jax.lax.cond((c < 0.0) & (B < 0.0), lambda: 3, lambda: 0)
        optim_case = _maybe_update_case(optim_case, (c < 0.0) & (B >= 0.0), 2, 0)
        optim_case = _maybe_update_case(optim_case, (c >= 0.0) & (B >= 0.0), 1, 0)
        return optim_case, w, r, s, A, B

    if safe:
        optim_case, w, r, s, A, B = jax.lax.cond(
            jax.lax.bitwise_and(jnp.dot(b, b) <= 1e-8, c < 0.0), trpo, cpo
        )
    else:
        optim_case, w, r, s, A, B = trpo()

    def no_recovery():
        def feasible_cases():
            lam = jnp.sqrt(q / (2.0 * target_kl))
            nu = 0.0
            return lam, nu

        def non_feasible_cases():
            LA, LB = [0.0, r / c], [r / c, np.inf]
            LA, LB = jax.lax.cond(c < 0, lambda: (LA, LB), lambda: (LB, LA))
            proj = lambda x, L: jnp.maximum(L[0], jnp.minimum(L[1], x))
            lam_a = proj(jnp.sqrt(A / (B + 1e-8)), LA)
            lam_b = proj(jnp.sqrt(q / (2 * target_kl)), LB)
            f_a = lambda lam: -0.5 * (A / (lam + 1e-8) + B * lam) - r * c / (s + 1e-8)
            f_b = lambda lam: -0.5 * (q / (lam + 1e-8) + 2.0 * target_kl * lam)
            lam = jnp.where(f_a(lam_a) >= f_b(lam_b), lam_a, lam_b)
            nu = jnp.maximum(0, lam * c - r) / (s + 1e-8)
            return lam, nu

        return jax.lax.cond(optim_case > 2, feasible_cases, non_feasible_cases)

    def recovery():
        lam = 0.0
        nu = jnp.sqrt(2.0 * target_kl / (s + 1e-8))
        return lam, nu

    lam, nu = jax.lax.cond(optim_case == 0, recovery, no_recovery)
    direction = jax.lax.cond(
        optim_case > 0, lambda: (v + nu * w) / (lam + 1e-8), lambda: nu * w
    )
    return direction, optim_case


def backtracking(
    direction: jnp.ndarray,
    evaluate_policy: Callable,
    old_pi_loss: jnp.ndarray,
    old_surrogate_cost: jnp.ndarray,
    optim_case: int,
    c: jnp.ndarray,
    old_params: hk.Params,
    safe: bool,
    backtrack_iters: int,
    backtrack_coeff: float,
    target_kl: float,
):
    def cond(val):
        iter_, _, info = val
        kl = info["agent/actor/delta_kl"]
        new_pi_loss = info["agent/actor/new_pi_loss"]
        new_surrogate_cost = info["agent/actor/new_surrogate_cost"]
        loss_improve = jax.lax.cond(
            optim_case > 1, lambda: new_pi_loss <= old_pi_loss, lambda: True
        )
        if safe:
            cost_improve = new_surrogate_cost - old_surrogate_cost <= jnp.maximum(-c, 0)
        else:
            cost_improve = True
        kl_cond = kl <= target_kl
        improve = loss_improve & cost_improve & kl_cond
        return (~improve) & (iter_ < backtrack_iters)

    def body(val):
        iter_, *_ = val
        step_size = backtrack_coeff**iter_
        p, unravel_params = jax.flatten_util.ravel_pytree(old_params)
        new_params = unravel_params(p - step_size * direction)
        new_pi_loss, new_surrogate_cost, kl_d = evaluate_policy(new_params)
        return (
            iter_ + 1,
            new_params,
            {
                "agent/actor/delta_kl": kl_d,
                "agent/actor/new_pi_loss": new_pi_loss,
                "agent/actor/new_surrogate_cost": new_surrogate_cost,
            },
        )

    init_state = (
        0,
        old_params,
        {
            "agent/actor/delta_kl": target_kl + 1e-8,
            "agent/actor/new_pi_loss": old_pi_loss + 1e-8,
            "agent/actor/new_surrogate_cost": old_surrogate_cost - 1e-8,
        },
    )
    iters, new_actor_params, info = jax.lax.while_loop(cond, body, init_state)
    # If used all backtracking iterations, fall back to the old policy.
    new_actor_params = jax.lax.cond(
        iters == backtrack_iters, lambda: old_params, lambda: new_actor_params
    )
    info["agent/actor/line_search_step"] = iters
    info["agent/actor/optim_case"] = optim_case
    return new_actor_params, info


# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def _maybe_update_case(optim_case, pred, true_val, false_val):
    return jax.lax.cond(
        optim_case != 0,
        lambda: optim_case,
        lambda: jax.lax.cond(pred, lambda: true_val, lambda: false_val),
    )
