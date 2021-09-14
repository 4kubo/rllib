"""Implementation of a Linear Dynamical System."""

import numpy as np
import torch
from gym import spaces

from .abstract_system import AbstractSystem


class LinearSystem(AbstractSystem):
    """An environment Discrete Time for Linear Dynamical Systems.

    Parameters
    ----------
    a: ndarray
        state transition matrix.
    b: ndarray
        input matrix.
    c: ndarray, optional
        observation matrix.
    """

    _action_scale = 0.1
    _state_scale = 2.0

    def __init__(self, a, b, c=None):
        self.a = torch.atleast_2d(a)
        self.b = torch.atleast_2d(b)
        if c is None:
            c = torch.eye(self.a.shape[0])
        self.c = c

        dim_state, dim_action = self.b.shape
        dim_observation = self.c.shape[0]

        super().__init__(
            dim_state=dim_state, dim_action=dim_action, dim_observation=dim_observation
        )
        self._state = None

    def step(self, action):
        """See `AbstractSystem.step'."""
        action = torch.atleast_2d(action)
        self.state = torch.clamp(self.state @ self.a + action @ self.b.T, -2.0, 2.0)
        return torch.clamp((self.state @ self.c).squeeze(0), -2.0, 2.0)

    def reset(self, state):
        """See `AbstractSystem.reset'."""
        self._time = 0
        self.state = torch.tensor(state, dtype=torch.get_default_dtype())
        # return (self.state @ self.c).numpy()
        return self.state @ self.c

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def action_space(self):
        """Return action space."""
        return spaces.Box(
            np.array([-self._action_scale] * self.dim_action),
            np.array([self._action_scale] * self.dim_action),
        )

    @property
    def observation_space(self):
        """Return observation space."""
        return spaces.Box(
            np.array([-self._state_scale] * self.dim_observation),
            np.array([self._state_scale] * self.dim_observation),
        )
