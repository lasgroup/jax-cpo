import logging
import os
import warnings
if 'LOG' not in os.environ:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  logging.getLogger().setLevel('ERROR')
  warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  warnings.filterwarnings("ignore", category=FutureWarning)

from copy import deepcopy
from types import SimpleNamespace
from gym import spaces
import haiku as hk

from jax_cpo import logging
from jax_cpo import models
from jax_cpo import cpo


def make(config: SimpleNamespace, observation_space: spaces.Space,
         action_space: spaces.Space, logger: logging.TrainingLogger):
  actor = hk.without_apply_rng(
      hk.transform(lambda x: models.Actor(
          **config.actor, output_size=action_space.shape)(x)))
  critic = hk.without_apply_rng(
      hk.transform(
          lambda x: models.DenseDecoder(**config.critic, output_size=(1,))(x)))
  safety_critic = deepcopy(critic)
  return cpo.CPO(observation_space, action_space, config, logger, actor, critic,
                 safety_critic)
