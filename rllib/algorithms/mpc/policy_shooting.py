"""MPC Algorithms."""

from rllib.util.value_estimation import mb_return

from .random_shooting import RandomShooting


class PolicyShooting(RandomShooting):
    r"""Policy shooting the MPC problem by sampling from a policy.

    Parameters
    ----------
    policy: AbstractPolicy.

    Other Parameters
    ----------------
    See Also: RandomShooting.

    References
    ----------
    Hong, Z. W., Pajarinen, J., & Peters, J. (2019).
    Model-based lookahead reinforcement learning. arXiv.
    """

    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def forward(self, state, **kwargs):
        """Get best action."""
        if self.training:
            returns, trajectory = mb_return(
                state,
                dynamical_model=self.dynamical_model,
                reward_model=self.reward_model,
                policy=self.policy,
                num_steps=self.horizon,
                gamma=self.gamma,
                num_samples=self.num_samples,
                value_function=self.terminal_reward,
                termination_model=self.termination_model,
                reduction="mean",
            )
            returns = returns.reshape((-1, self.num_samples)).squeeze(0)
            actions = trajectory.action

            # Reshape to match with `get_best_action` method
            actions = actions.reshape(
                (-1, self.num_samples, self.horizon, *self.dynamical_model.dim_action)
            )
            permute_dim = actions.dim() - len(self.dynamical_model.dim_action)
            permute_shape = (permute_dim - 1, *list(range(permute_dim - 1)), -1)
            actions = actions.permute(*permute_shape)
            elite_actions = self.get_best_action(actions, returns).mean(-2)

            # Return first action and the mean over the elite samples.
            return elite_actions, returns, trajectory
        else:
            elite_actions = self.policy(state)[0].unsqueeze(0)
            return elite_actions, None, None
