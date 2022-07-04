from typing import Tuple, NamedTuple

import numpy as np

from jax_cpo import transition as t


class TrajectoryData(NamedTuple):
  o: np.ndarray
  a: np.ndarray
  r: np.ndarray
  c: np.ndarray


class EpisodicTrajectoryBuffer:

  def __init__(self, batch_size: int, max_length: int, observation_shape: Tuple,
               action_shape: Tuple):
    self.idx = 0
    self.episode_id = 0
    self._full = False
    self.observation = np.zeros(
        (
            batch_size,
            max_length + 1,
        ) + observation_shape, dtype=np.float32)
    self.action = np.zeros(
        (
            batch_size,
            max_length,
        ) + action_shape, dtype=np.float32)
    self.reward = np.zeros((
        batch_size,
        max_length,
    ), dtype=np.float32)
    self.cost = np.zeros((
        batch_size,
        max_length,
    ), dtype=np.float32)

  def add(self, transition: t.Transition):
    """
    Adds transitions to the current running trajectory.
    """
    batch_size = min(transition.observation.shape[0], self.observation.shape[1])
    episode_slice = slice(self.episode_id, self.episode_id + batch_size)
    self.observation[episode_slice,
                     self.idx] = transition.observation[:batch_size].copy()
    self.action[episode_slice, self.idx] = transition.action[:batch_size].copy()
    self.reward[episode_slice, self.idx] = transition.reward[:batch_size].copy()
    self.cost[episode_slice, self.idx] = transition.cost[:batch_size].copy()
    if transition.last:
      assert self.idx == self.reward.shape[2] - 1
      self.observation[episode_slice, self.idx +
                       1] = transition.next_observation[:batch_size].copy()
      if self.episode_id + batch_size == self.observation.shape[1]:
        self._full = True
      else:
        self.idx = -1
      self.episode_id += batch_size
    self.idx += 1

  def dump(self) -> TrajectoryData:
    """
    Returns all trajectories from all tasks (with shape [N_tasks, K_episodes,
    T_steps, ...]).
    """
    o = self.observation
    a = self.action
    r = self.reward
    c = self.cost
    # Reset the on-policy running cost.
    self.idx = 0
    self.episode_id = 0
    self._full = False
    return TrajectoryData(o, a, r, c)

  @property
  def full(self):
    return self._full
