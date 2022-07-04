from collections import defaultdict
from typing import Callable, Optional, Dict, List, DefaultDict

import numpy as np
from tqdm import tqdm

from jax_cpo import cpo
from jax_cpo import transition as t
from jax_cpo import episodic_async_env as esye

EpisodeSummary = Dict[str, List]
IterationSummary = List[EpisodeSummary]


def interact(agent: cpo.CPO,
             environment: esye.EpisodicAsync,
             steps: int,
             train: bool,
             on_episode_end: Optional[Callable[[EpisodeSummary, bool],
                                               None]] = None,
             render_episodes: int = 0,
             render_mode: str = 'rgb_array') -> [IterationSummary]:
  observations = environment.reset()
  step = 0
  episodes = [defaultdict(list, {'observation': [observations]})]
  # Discard transitions from environments such that episodes always finish
  # after time_limit
  discard = min(steps // environment.time_limit, environment.num_envs)
  with tqdm(total=steps) as pbar:
    while step < steps:
      if render_episodes:
        frames = environment.render(render_mode)
        episodes[-1]['frames'].append(frames)
      actions = agent(observations, train)
      next_observations, rewards, dones, infos = environment.step(actions)
      costs = np.array([info.get('cost', 0) for info in infos])
      transition = t.Transition(
          *map(lambda x: x[:discard], (observations, next_observations, actions,
                                       rewards, costs, dones, infos)))
      episodes[-1] = _append(transition, episodes[-1])
      if train:
        agent.observe(transition)
      observations = next_observations
      if transition.last:
        render_episodes = max(render_episodes - 1, 0)
        if on_episode_end:
          on_episode_end(episodes[-1])
        observations = environment.reset()
        episodes.append(defaultdict(list, {'observation': [observations]}))
      transition_steps = sum(transition.steps)
      step += transition_steps
      pbar.update(transition_steps)
    if not episodes[-1] or len(episodes[-1]['reward']) == 0:
      episodes.pop()
  return episodes


def _append(transition: t.Transition, episode: DefaultDict) -> DefaultDict:
  episode['observation'].append(transition.observation)
  episode['action'].append(transition.action)
  episode['reward'].append(transition.reward)
  episode['cost'].append(transition.cost)
  episode['done'].append(transition.done)
  episode['info'].append(transition.info)
  return episode
