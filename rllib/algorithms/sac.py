"""Soft Actor-Critic Algorithm."""
import torch

from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction
from .abstract_algorithm import AbstractAlgorithm


class SAC(AbstractAlgorithm):
    r"""Implementation of Soft Actor-Critic algorithm.

    SAC is an off-policy policy gradient algorithm.

    epsilon: Learned temperature as a constraint.
    eta: Fixed regularization.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    Stochastic Actor. ICML.

    Haarnoja, T., Zhou, A., ... & Levine, S. (2018).
    Soft actor-critic algorithms and applications. arXiv.
    """

    def __init__(self, eta=0.2, entropy_regularization=False, *args, **kwargs):
        super().__init__(
            eta=eta, entropy_regularization=entropy_regularization, *args, **kwargs
        )
        self.eta = eta
        assert (
            len(self.policy.dim_action) == 1
        ), "Only Nx1 continuous actions implemented."

    def post_init(self):
        """Set derived modules after initialization."""
        super().post_init()
        self.policy.dist_params.update(tanh=True)
        self.policy_target.dist_params.update(tanh=True)

    def actor_loss(self, observation):
        """Get Actor Loss."""
        assert isinstance(self.critic, NNEnsembleQFunction)
        state = observation.state
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        action_normalized = pi.rsample()
        log_p = pi.log_prob(action_normalized)

        action = self.policy.action_scale * action_normalized
        with DisableGradient(self.critic):
            q = self.critic(state, action)
            q = torch.min(q, -1).values
        # Take mean over time coordinate.
        if 1 < q.dim():
            q = q.mean(dim=1)
            log_p = log_p.mean(dim=1)
        actor_loss = -q + self.entropy_loss.eta * log_p

        return Loss(policy_loss=actor_loss)

    def critic_loss(self, observation):
        """Get critic loss.

        This is usually computed using fitted value iteration and semi-gradients.
        critic_loss = criterion(pred_q, target_q.detach()).

        Parameters
        ----------
        observation: Observation.
            Sampled observations.
            It is of shape B x N x d, where:
                - B is the batch size
                - N is the N-step return
                - d is the dimension of the attribute.

        Returns
        -------
        loss: Loss.
            Loss with parameters loss, critic_loss, and td_error filled.
        """
        pred_q = self.get_value_prediction(observation)

        # Get target_q with semi-gradients.
        with torch.no_grad():
            target_q = self.get_value_target(observation)

            target_q = target_q.unsqueeze(-1).repeat_interleave(
                self.critic.num_heads, -1
            )

        critic_loss = self.criterion(pred_q, target_q)

        # Ensembles have last dimension as ensemble head; sum all ensembles.
        critic_loss = critic_loss.sum(-1)
        # Take mean over time coordinate.
        critic_loss = critic_loss.mean(-1)

        return Loss(critic_loss=critic_loss)

    def regularization_loss(self, observation, num_trajectories=1):
        state = observation.state
        pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
        action_normalized = pi.rsample()
        log_p = pi.log_prob(action_normalized)

        # Take mean over time coordinate.
        if 1 < log_p.dim():
            log_p = log_p.mean(dim=1)

        # Compute dual loss at the same time
        dual_loss = (
            self.entropy_loss._eta()
            * (-log_p - self.entropy_loss.target_entropy).detach()
        )

        entropy = -pi.log_prob(pi.sample([4])).mean().detach()
        self._info.update(
            eta=self.entropy_loss.eta,
            entropy=self._info["entropy"] + entropy,
        )
        return Loss(dual_loss=dual_loss)
