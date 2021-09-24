"""Model for quadratic reward."""

import torch

from rllib.model.abstract_model import AbstractModel
from rllib.util.neural_networks import torch_quadratic


class QuadraticReward(AbstractModel):
    """Quadratic Reward Function."""

    def __init__(self, q, r, goal=None):
        if goal is None:
            goal = torch.zeros(q.shape[-1])
        super().__init__(
            goal=goal,
            dim_state=(q.shape[-1],),
            dim_action=(r.shape[-1],),
            model_kind="rewards",
        )

        self.q = q
        self.r = r

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        device = state.device
        shape = state.shape
        state_cost = torch_quadratic(
            state.view(-1, self.dim_state[0]) - self.goal.to(device=device),
            self.q.to(device=device),
        )
        action = action.view(-1, self.dim_action[0])
        action_cost = torch_quadratic(
            action[:, : self.dim_action[0]], self.r.to(device=device)
        )
        return -(state_cost + action_cost).view(shape[:-1]).squeeze(-1), torch.zeros(1)
