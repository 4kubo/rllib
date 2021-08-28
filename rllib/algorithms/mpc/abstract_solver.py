"""MPC Algorithms."""
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from rllib.util.rollout import rollout_actions
from rllib.util.value_estimation import discount_sum


class MPCSolver(nn.Module, metaclass=ABCMeta):
    r"""Solve the discrete time trajectory optimization controller.

    ..math :: u[0:H-1] = \arg \max \sum_{t=0}^{H-1} r(x0, u) + final_reward(x_H)

    When called, it will return the sequence of actions that solves the problem.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    horizon: int.
        Horizon to solve planning problem.
    gamma: float, optional.
        Discount factor.
    scale: float, optional.
        Scale of covariance matrix to sample.
    num_mpc_iter: int, optional.
        Number of iterations of solver method.
    num_action_samples: int, optional.
        Number of samples for shooting method.
    termination_model: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.
    warm_start: bool, optional.
        Whether or not to start the optimization with a warm start.
    default_action: str, optional.
         Default action behavior.
    num_part: int, optional.
        The number of particles for taking averages of action's expected reward.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        horizon=25,
        gamma=1.0,
        num_mpc_iter=1,
        num_action_samples=400,
        termination_model=None,
        scale=0.3,
        terminal_reward=None,
        warm_start=True,
        clamp=True,
        default_action="zero",
        action_scale=1.0,
        num_part=20,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._submodules = {
            "dynamical_model": dynamical_model,
            "reward_model": reward_model,
            "termination_model": termination_model,
        }

        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.horizon = horizon
        self.gamma = gamma

        self.num_mpc_iter = num_mpc_iter
        self.num_samples = num_action_samples
        self.num_part = num_part

        self.terminal_reward = terminal_reward
        self.warm_start = warm_start
        self.default_action = default_action
        self.dim_action = self.dynamical_model.dim_action[0]

        self.mean = None
        self._scale = scale
        self.covariance = (scale ** 2) * torch.eye(self.dim_action).repeat(
            self.horizon, 1, 1
        )
        if isinstance(action_scale, np.ndarray):
            action_scale = torch.tensor(action_scale, dtype=torch.get_default_dtype())
        elif not isinstance(action_scale, torch.Tensor):
            action_scale = torch.full((self.dim_action,), action_scale)
        if len(action_scale) < self.dim_action:
            extra_dim = self.dim_action - len(action_scale)
            action_scale = torch.cat((action_scale, torch.ones(extra_dim)))

        self.register_buffer("action_scale", action_scale)
        self.clamp = clamp

    def evaluate_action_sequence(self, action_sequence, state):
        """Evaluate action sequence by performing a rollout.

        Parameters
        ----------
        action_sequence: Tensor
            Tensor of dimension [horizon, batch_size x num_samples, dim_act]
        state: Tensor
            Tensor of dimension [batch_size x num_samples, dim_state]
        Returns
        -------
        returns: Tensor
            Tensor of dimension [batch_size x num_samples]
        steps: list
            In length of horizon where each in batch_size x num_samples `Observation`
        """
        steps = rollout_actions(
            self.dynamical_model,
            self.reward_model,
            self.action_scale.data * action_sequence,  # scale actions.
            state,
            self.termination_model,
            device=self.device,
        )

        rewards = torch.stack(
            [step.reward.reshape(-1, self.num_part).mean(1) for step in steps], dim=1
        )
        returns = discount_sum(rewards, self.gamma, device=self.device)

        # TODO: Deal with the case for terminal_reward even if termination is not at `self.horizon`
        # if self.terminal_reward:
        #     terminal_reward = self.terminal_reward(observations.next_state[..., -1, :])
        #     returns = returns + self.gamma ** self.horizon * terminal_reward
        return returns, steps

    @abstractmethod
    def get_candidate_action_sequence(self):
        """Get candidate actions."""
        raise NotImplementedError

    @abstractmethod
    def get_best_action(self, action_sequence, returns):
        """Get best action."""
        raise NotImplementedError

    @abstractmethod
    def update_sequence_generation(self, elite_actions):
        """Update sequence generation."""
        raise NotImplementedError

    def initialize_actions(self, batch_shape):
        """Initialize mean and covariance of action distribution."""
        if self.warm_start and self.mean is not None:
            next_mean = self.mean[1:, ..., :]
            if self.default_action == "zero":
                final_action = torch.zeros_like(self.mean[:1, ..., :])
            elif self.default_action == "constant":
                final_action = self.mean[-1:, ..., :]
            elif self.default_action == "mean":
                final_action = torch.mean(next_mean, dim=0, keepdim=True)
            else:
                raise NotImplementedError
            self.mean = torch.cat((next_mean, final_action), dim=0).to(self.device)
        else:
            self.mean = torch.zeros(
                self.horizon, *batch_shape, self.dim_action, device=self.device
            )
        covariance = (self._scale ** 2) * torch.eye(self.dim_action).repeat(
            self.horizon, *batch_shape, 1, 1
        )
        self.covariance = covariance.to(self.device)

    def get_action_sequence_and_returns(
        self, state, action_sequence, returns, process_nr=0
    ):
        """Get action_sequence and returns associated.

        These are bundled for parallel execution.

        The data inside action_sequence and returns will get modified.
        """
        if self.num_cpu > 1:
            # Multi-Processing inherits random state.
            torch.manual_seed(int(1000 * time.time()))

        action_sequence[:] = self.get_candidate_action_sequence()
        returns[:] = self.evaluate_action_sequence(action_sequence, state)

    def forward(self, state):
        """Return action that solves the MPC problem."""
        batch_shape = state.shape[:-1]
        self.initialize_actions(batch_shape)

        state = state.repeat(self._repeat_shape)

        returns, steps = None, None
        for _ in range(self.num_mpc_iter):
            (
                action_candidates,
                action_candidates_eval,
            ) = self.get_candidate_action_sequence()
            returns, steps = self.evaluate_action_sequence(
                action_candidates_eval, state
            )
            elite_actions = self.get_best_action(action_candidates, returns)
            self.update_sequence_generation(elite_actions)

        if self.clamp:
            return self.mean.clamp(-1.0, 1.0), returns, steps

        return self.mean, returns, steps

    def reset(self, state=None, warm_action=None):
        """Reset warm action."""
        self.mean = warm_action
        self._repeat_shape = (self.num_samples * self.num_part, max(1, state.ndim - 1))
        # Set prediction strategy for trajectory sampling
        if state is not None:
            assert isinstance(state, np.ndarray)
            prop_type = self.dynamical_model.get_prediction_strategy()
            if state.ndim == 1:
                dims_sample = list(state.shape)
                sample_shape = [self.num_samples * self.num_part] + dims_sample
            else:
                n_batch = state.shape[0]
                dims_sample = list(state.shape[1:])
                sample_shape = [
                    n_batch * self.num_samples * self.num_part
                ] + dims_sample
            self.dynamical_model.set_prediction_strategy(prop_type, sample_shape)
            self.reward_model.set_prediction_strategy(prop_type, sample_shape)

    @property
    def device(self):
        if hasattr(self, "_device"):
            return self._device
        else:
            self._device = self.action_scale.device
            return self._device

    @property
    def dynamical_model(self):
        return self._submodules["dynamical_model"]

    @property
    def reward_model(self):
        return self._submodules["reward_model"]

    @property
    def termination_model(self):
        return self._submodules["termination_model"]
