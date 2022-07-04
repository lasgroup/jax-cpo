from typing import NamedTuple, Tuple, Dict

import numpy as np


class Transition(NamedTuple):
  observation: np.ndarray
  next_observation: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  cost: np.ndarray
  done: np.ndarray
  info: Tuple[Dict]

  @property
  def last(self):
    return self.done.all()

  @property
  def first(self):
    return all(info.get('first', False) for info in self.info)

  @property
  def steps(self):
    return [info.get('steps', 1) for info in self.info]
