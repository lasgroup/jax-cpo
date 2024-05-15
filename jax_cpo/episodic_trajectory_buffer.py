import numpy as np

from jax_cpo.rl.trajectory import TrajectoryData


class EpisodicTrajectoryBuffer:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        observation_shape: tuple[int],
        action_shape: tuple[int],
    ):
        self.episode_id = 0
        self._full = False
        self.dtype = np.float32
        self.observation = np.zeros(
            (
                batch_size,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.dtype,
        )
        self.action = np.zeros(
            (
                batch_size,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (
                batch_size,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                batch_size,
                max_length,
            ),
            dtype=self.dtype,
        )

    def add(self, trajectory: TrajectoryData):
        capacity, _ = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.episode_id + batch_size, capacity)
        episode_slice = slice(self.episode_id, end)
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[episode_slice] = val[:batch_size].astype(self.dtype)
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size],
                trajectory.next_observation[:batch_size, -1:],
            ],
            axis=1,
        )
        self.observation[episode_slice] = observation.astype(self.dtype)
        self.episode_id = self.episode_id + batch_size
        if self.episode_id >= capacity:
            self._full = True

    def dump(self) -> TrajectoryData:
        """
        Returns all trajectories from all tasks (with shape [K_episodes,
        T_steps, ...]).
        """
        o = self.observation[:, :-1]
        next_o = self.observation[:, 1:]
        a = self.action
        r = self.reward
        c = self.cost
        # Reset the on-policy running cost.
        self.episode_id = 0
        self._full = False
        return TrajectoryData(o, next_o, a, r, c)

    @property
    def full(self):
        return self._full
