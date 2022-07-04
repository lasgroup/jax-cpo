import os
import pytest

from jax_cpo import cpo
from jax_cpo import config as options
from jax_cpo import trainer


@pytest.mark.not_safe
def test_not_safe():

  def make_env(config):
    import gym
    env = gym.make('HalfCheetah-v2')
    env._max_episode_steps = config.time_limit
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'cpo',
      '--num_trajectories', '300', '--time_limit', '150', '--vf_iters', '10',
      '--eval_trials', '0', '--train_driver.adaptation_steps', '45000',
      '--render_episodes', '0', '--test_driver.adaptation_steps', '1500',
      '--lambda_', '0.95', '--epochs', '100', '--safe', 'False', '--log_dir',
      'results/test_cpo_not_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else trainer.Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] > 115.
  assert constraint[config.task] == 0.