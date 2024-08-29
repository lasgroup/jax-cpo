from gymnasium.wrappers.compatibility import EnvCompatibility
from omegaconf import DictConfig
import numpy as np
from jax_cpo.benchmark_suites.utils import get_domain_and_task

from jax_cpo.rl.types import EnvironmentFactory


def sample_task(seed):
    easy_tasks = (
        "go_to_goal",
        "push_box",
        "collect",
        "catch_goal",
        "press_buttons",
        "unsupervised",
    )
    task = np.random.RandomState(seed).choice(easy_tasks)
    return task


class SafeAdaptationEnvCompatibility(EnvCompatibility):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset(options=options), {}


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import safe_adaptation_gym

        _, task_cfg = get_domain_and_task(cfg)
        if task_cfg.task is not None:
            task = task_cfg.task
        else:
            task = sample_task(cfg.training.seed)
        env = safe_adaptation_gym.make(
            robot_name=task_cfg.robot_name,
            task_name=task,
            seed=cfg.training.seed,
            render_lidar_and_collision=False,
        )
        env = SafeAdaptationEnvCompatibility(env)
        return env

    return make_env  # type: ignore
