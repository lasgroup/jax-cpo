import os
import hydra
from jax_cpo import trainer as t

from copy import deepcopy
from types import SimpleNamespace
from gym import spaces
import haiku as hk

from jax_cpo import logging
from jax_cpo import models
from jax_cpo import cpo


def make(
    config: SimpleNamespace,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    logger: logging.TrainingLogger,
):
    actor = hk.without_apply_rng(
        hk.transform(
            lambda x: models.Actor(**config.actor, output_size=action_space.shape)(x)
        )
    )
    critic = hk.without_apply_rng(
        hk.transform(
            lambda x: models.DenseDecoder(**config.critic, output_size=(1,))(x)
        )
    )
    safety_critic = deepcopy(critic)
    return cpo.CPO(
        observation_space, action_space, config, logger, actor, critic, safety_critic
    )


@hydra.main(version_base=None, config_path="jax_cpo", config_name="configs")
def main(config):
    def make_env(config):
        import safe_adaptation_gym
        from gym.wrappers.compatibility import EnvCompatibility

        env = safe_adaptation_gym.make(config.robot, config.task)
        env = EnvCompatibility(env)
        return env

    if not config.jit:
        from jax.config import config as jax_config

        jax_config.update("jax_disable_jit", True)
    path = os.path.join(config.log_dir, "state.pkl")
    with t.Trainer.from_pickle(config) if os.path.exists(path) else t.Trainer(
        config=config, make_agent=make, make_env=lambda: make_env(config)
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
