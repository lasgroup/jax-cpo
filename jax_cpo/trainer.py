import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Optional, List, Dict, Callable

import cloudpickle
import numpy as np
from gym import Env
from gym.spaces import Space

from jax_cpo import cpo
from jax_cpo import episode_sampling as es
from jax_cpo import episodic_async_env
from jax_cpo import logging


def evaluation_summary(episodes: es.IterationSummary,
                       prefix: str = 'evaluation') -> Dict:
  summary = defaultdict(float)
  return_ = lambda arr: np.asarray(arr).sum(0).mean()
  summary[f'{prefix}/reward_return'] = return_(
      [episode['reward'] for episode in episodes])
  summary[f'{prefix}/cost_return'] = return_(
      [episode['cost'] for episode in episodes])
  if frames := episodes[0].get('frames', []):
    summary[f'{prefix}/frames'] = frames
  return summary


def on_episode_end(episode: es.EpisodeSummary, logger: logging.TrainingLogger,
                   train: bool):

  def return_(arr):
    return np.asarray(arr).sum(0).mean()

  episode_return = return_(episode['reward'])
  cost_return = return_(episode['cost'])
  print("\nreward return: {:.4f} / cost return: {:.4f}".format(
      episode_return, cost_return))
  if train:
    summary = {
        f'training/episode_return': episode_return,
        f'training/episode_cost_return': cost_return
    }
    logger.log_summary(summary)
    logger.step += np.asarray(episode['reward']).size


def log_videos(logger: logging.TrainingLogger, videos: List, epoch: int):
  logger.log_video(
      np.asarray(videos).transpose([1, 0, 2, 3, 4])[:1],
      'evaluation/video',
      step=epoch)


class Trainer:

  def __init__(self,
               config: SimpleNamespace,
               make_env: Callable[[], Env],
               make_agent: Optional[Callable[
                   [SimpleNamespace, Space, Space, logging.TrainingLogger],
                   cpo.CPO]] = None,
               agent: Optional[cpo.CPO] = None,
               start_epoch: int = 0,
               seeds: Optional[List[int]] = None):
    self.config = config
    assert not (agent is not None and make_agent is not None), (
        'agent and make_agent parameters are mutually exclusive.')
    self.make_agent = make_agent
    self.agent = agent
    self.make_env = make_env
    self.epoch = start_epoch
    self.seeds = seeds
    self.logger = None
    self.state_writer = None
    self.env = None

  def __enter__(self):
    self.state_writer = logging.StateWriter(self.config.log_dir)
    self.logger = logging.TrainingLogger(self.config.log_dir)
    self.env = episodic_async_env.EpisodicAsync(self.make_env,
                                                self.config.parallel_envs,
                                                self.config.time_limit)
    if self.seeds is not None:
      self.env.reset(seed=self.seeds)
    else:
      self.env.reset(seed=self.config.seed)
    if self.make_agent is not None:
      self.agent = self.make_agent(self.config, self.env.observation_space,
                                   self.env.action_space, self.logger)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.state_writer.close()
    self.logger.close()

  def train(self, epochs: Optional[int] = None) -> [float, float]:
    config, agent, env = self.config, self.agent, self.env
    epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
    objective, constraint = defaultdict(float), defaultdict(float)
    for epoch in range(epoch, epochs or config.epochs):
      print('Training epoch #{}'.format(epoch))
      results = es.interact(
          agent, env, self.config.train_steps_per_epoch, True,
          lambda episode: on_episode_end(episode, logger, True))
      summary = evaluation_summary(results, 'on_policy_evaluation')
      logger.log_summary(summary, epoch)
      if epoch % config.eval_every == 0:
        print('Evaluating...')
        results = es.interact(
            self.agent, self.env, self.config.test_steps_per_epoch, False,
            lambda episode: on_episode_end(episode, self.logger, True),
            self.config.render_episodes)
        summary = evaluation_summary(results)
        logger.log_summary(summary, epoch)
        if videos := summary.get('evaluation/frames', []):
          log_videos(logger, videos, epochs)
      self.epoch = epoch + 1
      state_writer.write(self.state)
    logger.flush()
    return objective, constraint

  def get_env_random_state(self):
    rs = [
        state.get_state()[1]
        for state in self.env.get_attr('rs')
        if state is not None
    ]
    if not rs:
      rs = [
          state.bit_generator.state['state']['state']
          for state in self.env.get_attr('np_random')
      ]
    return rs

  @classmethod
  def from_pickle(cls, config: SimpleNamespace):
    with open(os.path.join(config.log_dir, 'state.pkl'), 'rb') as f:
      make_env, env_rs, agent, epoch = cloudpickle.load(f).values()
    print('Resuming experiment from: {}...'.format(config.log_dir))
    assert agent.config == config, 'Loaded different hyperparameters.'
    return cls(
        config=agent.config,
        make_env=make_env,
        start_epoch=epoch,
        seeds=env_rs,
        agent=agent)

  @property
  def state(self):
    return {
        'make_env': self.make_env,
        'env_rs': self.get_env_random_state(),
        'agent': self.agent,
        'epoch': self.epoch
    }
