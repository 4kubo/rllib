"""Implementation of a LQR environment."""

import numpy as np
import torch

from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import LinearSystem
from rllib.reward.quadratic_reward import QuadraticReward


class LQREnv(SystemEnvironment):
    """An environment of linear dynamical system and quadratic reward function."""

    def __init__(
        self, dim_state, dim_action=None, ctrl_cost_weight=1.0, random_init=False
    ):
        # System
        if dim_action is None:
            dim_action = dim_state
        a = torch.eye(dim_state) * (1.0 + 1 / np.sqrt(dim_state) * 0.1)
        b = torch.zeros((dim_state, dim_action))
        index = torch.tensor([[min(b.shape[1] - 1, i)] for i in range(b.shape[0])])
        b = torch.scatter(b, dim=1, index=index, value=1 / np.sqrt(dim_state))
        linear_system = LinearSystem(a, b)

        if random_init:

            def initial_state():
                state_init = np.random.randn((dim_state))
                state_init /= np.linalg.norm(state_init)
                return state_init

        else:
            initial_state = np.ones(dim_state) / np.sqrt(dim_state)

        # Reward function
        q = torch.eye(dim_state, dtype=torch.float32)
        r = torch.eye(dim_action, dtype=torch.float32) * ctrl_cost_weight
        reward_model = QuadraticReward(q, r)
        super(LQREnv, self).__init__(
            linear_system,
            initial_state=initial_state,
            reward=reward_model,
        )

        self.viewer = None

        # Rendering
        self._screen_width = 500
        self._screen_unit = 200

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self._screen_width, self._screen_width)
            # Controlled point
            self.circle_trans = rendering.Transform()
            circle = rendering.make_circle(self._screen_width / 100)
            circle.set_color(0.8, 0.6, 0.4)
            circle.add_attr(self.circle_trans)
            self.viewer.add_geom(circle)
            #
            x_coordinate = rendering.Line(
                (0, self._screen_width / 2),
                (self._screen_width, self._screen_width / 2),
            )

            y_coordinate = rendering.Line(
                (self._screen_width / 2, 0),
                (self._screen_width / 2, self._screen_width),
            )
            self.viewer.add_geom(x_coordinate)
            self.viewer.add_geom(y_coordinate)

        x, y = torch.atleast_2d(self.system.state)[0, :2]
        self.circle_trans.set_translation(
            x * self._screen_unit + self._screen_width / 2,
            y * self._screen_unit + self._screen_width / 2,
        )
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.cloose()
            self.viewer = None

    def reward_model(self):
        return self.reward
