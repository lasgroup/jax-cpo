import logging
import os
import time
from typing import Optional

import cloudpickle
from gymnasium import Space
from omegaconf import DictConfig

from jax_cpo.rl import acting, episodic_async_env
from jax_cpo.rl.epoch_summary import EpochSummary
from jax_cpo.rl.logging import StateWriter, TrainingLogger
from jax_cpo.rl.types import Agent, EnvironmentFactory
from jax_cpo.rl.utils import PRNGSequence
import haiku as hk
from copy import deepcopy

from jax_cpo import models
from jax_cpo import cpo

_LOG = logging.getLogger(__name__)

_TRAINING_STATE = "state.pkl"


def get_state_path() -> str:
    log_path = os.getcwd()
    state_path = os.path.join(log_path, _TRAINING_STATE)
    return state_path


def should_resume(state_path: str) -> bool:
    return os.path.exists(state_path)


def start_fresh(
    cfg: DictConfig,
    make_env: EnvironmentFactory,
) -> "Trainer":
    return Trainer(cfg, make_env)


def load_state(cfg, state_path) -> "Trainer":
    return Trainer.from_pickle(cfg, state_path)


def make_agent(
    config: DictConfig,
    observation_space: Space,
    action_space: Space,
):
    actor = hk.without_apply_rng(
        hk.transform(
            lambda x: models.Actor(**config.agent.actor, output_size=action_space.shape)(x)
        )
    )
    critic = hk.without_apply_rng(
        hk.transform(
            lambda x: models.DenseDecoder(**config.agent.critic, output_size=(1,))(x)
        )
    )
    safety_critic = deepcopy(critic)
    return cpo.CPO(
        observation_space, action_space, config, actor, critic, safety_critic
    )


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: Agent | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        self.config = config
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: TrainingLogger | None = None
        self.state_writer: StateWriter | None = None
        self.env: episodic_async_env.EpisodicAsync | None = None
        self.agent = agent

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = TrainingLogger(self.config)
        self.state_writer = StateWriter(log_path, _TRAINING_STATE)
        self.env = episodic_async_env.EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        if self.seeds is None:
            self.seeds = PRNGSequence(self.config.training.seed)
        if self.agent is None:
            self.agent = make_agent(
                self.config, self.env.observation_space, self.env.action_space
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer, agent = (
            self.epoch,
            self.logger,
            self.state_writer,
            self.agent,
        )
        assert logger is not None and state_writer is not None and agent is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            _LOG.info(f"Training epoch #{epoch}")
            summary, wall_time, steps = self._run_training_epoch(
                self.config.training.episodes_per_epoch
            )
            objective, cost_rate, feasibilty = summary.metrics
            metrics = {
                "train/objective": objective,
                "train/cost_rate": cost_rate,
                "train/feasibility": feasibilty,
                "train/fps": steps / wall_time,
            }
            report = agent.report(summary, epoch, self.step)
            report.metrics.update(metrics)
            if (maybe_videos := summary.videos) is not None:
                report.videos.update({"train/video": maybe_videos})
            logger.log(report.metrics, self.step)
            for k, v in report.videos.items():
                logger.log_video(v, self.step, k)
            self.epoch = epoch + 1
            state_writer.write(self.state)

    def _run_training_epoch(
        self,
        episodes_per_epoch: int,
    ) -> tuple[EpochSummary, float, int]:
        agent, env, logger, seeds = self.agent, self.env, self.logger, self.seeds
        assert (
            env is not None
            and agent is not None
            and logger is not None
            and seeds is not None
        )
        start_time = time.time()
        env.reset(seed=int(next(seeds)[0].item()))
        summary, step = acting.epoch(
            agent,
            env,
            episodes_per_epoch,
            True,
            self.step,
            self.config.training.render_episodes,
        )
        steps = step - self.step
        self.step = step
        next(seeds)
        end_time = time.time()
        wall_time = end_time - start_time
        return summary, wall_time, steps

    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "Trainer":
        with open(state_path, "rb") as f:
            make_env, seeds, agent, epoch, step = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        _LOG.info(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=seeds,
            agent=agent,
            step=step,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "seeds": self.seeds,
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }
