from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
from numpy import typing as npt

from jax_cpo.rl.trajectory import Trajectory


@dataclass
class EpochSummary:
    _data: list[list[Trajectory]] = field(default_factory=list)
    cost_boundary: float = 25.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> Tuple[float, float]:
        rewards, costs = [], []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                *_, r, c = trajectory.as_numpy()
                rewards.append(r)
                costs.append(c)
        # Stack data from all tasks on the first axis,
        # giving a [#tasks, #episodes, #time, ...] shape.
        stacked_rewards = np.stack(rewards)
        stacked_costs = np.stack(costs)
        return (
            _objective(stacked_rewards),
            _objective(stacked_costs),
        )

    @property
    def videos(self):
        all_vids = []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                if len(trajectory.frames) > 0:
                    all_vids.append(trajectory.frames)
        if len(all_vids) == 0:
            return None
        vids = np.asarray(all_vids)[-1].transpose(1, 0, -1, 2, 3)
        return vids

    def extend(self, samples: List[Trajectory]) -> None:
        self._data.append(samples)


def _objective(rewards: npt.NDArray[Any]) -> float:
    return float(rewards.sum(2).mean())


def _feasibility(costs: npt.NDArray[Any], boundary: float) -> float:
    return float((costs.sum(2).mean(1) <= boundary).mean())
