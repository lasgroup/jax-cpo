import os
import pytest

import jax_cpo
from jax_cpo import config as options
from jax_cpo import trainer as t


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
  with t.Trainer.from_pickle(config) if os.path.exists(path) else t.Trainer(
      config=config, make_agent=jax_cpo.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] > 115.
  assert constraint[config.task] == 0.


@pytest.mark.safe
def test_safe():

  def make_env(config):
    import safe_adaptation_gym
    env = safe_adaptation_gym.make(
        config.robot,
        config.task,
        config={
            'obstacles_size_noise_scale': 0.,
            'robot_ctrl_range_scale': 0.
        })
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'cpo',
      '--num_trajectories', '30', '--eval_trials', '1', '--render_episodes',
      '0', '--train_driver.adaptation_steps', '30000', '--epochs', '334',
      '--safe', 'True', '--log_dir', 'results/test_cpo_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with t.Trainer.from_pickle(config) if os.path.exists(path) else t.Trainer(
      config=config, make_agent=jax_cpo.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 7.
  assert constraint[config.task] < config.cost_limit
