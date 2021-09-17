"""Implementation of DQNAgent Algorithms."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sac import SAC
from rllib.dataset.datatypes import Loss
from rllib.policy import NNPolicy
from rllib.util.neural_networks import DisableGradient
from rllib.util.parameter_decay import Constant, Learnable
from rllib.value_function import NNEnsembleQFunction
from .off_policy_agent import OffPolicyAgent


class SACAgent(OffPolicyAgent):
    """Implementation of a SAC agent.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a
    stochastic actor. ICML.

    """

    def __init__(
        self,
        critic,
        policy,
        actor_optimizer=None,
        critic_optimizer=None,
        temp_optimizer=None,
        criterion=loss.MSELoss,
        eta=0.2,
        entropy_regularization=False,
        num_iter=50,
        train_frequency=50,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_iter=num_iter, train_frequency=train_frequency, *args, **kwargs
        )
        self.algorithm = SAC(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            eta=eta,
            entropy_regularization=entropy_regularization,
            *args,
            **kwargs,
        )

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temp_optimizer = temp_optimizer
        self.policy = self.algorithm.policy

    def custom_opt_step(self):
        self.algorithm.reset_info()

        device = self.device if self.device == "cpu" else None
        observation, idx, weight = self.memory.sample_batch(
            self.batch_size, device=device
        )
        with DisableGradient(self.algorithm.policy):
            self.critic_optimizer.zero_grad()
            critic_loss = self.algorithm.critic_loss(observation)
            loss = (critic_loss.critic_loss * weight.detach()).mean()
            loss.backward()
            self.critic_optimizer.step()

        with DisableGradient(self.algorithm.critic):
            self.actor_optimizer.zero_grad()
            actor_loss = self.algorithm.actor_loss(observation)
            loss = (actor_loss.policy_loss * weight.detach()).mean()
            loss.backward()
            self.actor_optimizer.step()

        if not isinstance(self.algorithm.eta, Constant):
            self.temp_optimizer.zero_grad()
            temp_loss = self.algorithm.regularization_loss(observation)
            loss = (temp_loss.dual_loss * weight.detach()).mean()
            loss.backward()
        else:
            tmep_loss = Loss()

        self.temp_optimizer.step()

        # Update memory
        self.memory.update(idx, critic_loss.td_error.abs().detach())

        return actor_loss + critic_loss + temp_loss

    @classmethod
    def default(
        cls,
        environment,
        policy=None,
        critic=None,
        lr=3e-4,
        fixed_temperature=False,
        layers=None,
        non_linearity="ReLU",
        eta=1.0,
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        layers = layers or [256, 256]
        if critic is None:
            critic = NNEnsembleQFunction.default(
                environment,
                layers=layers,
                non_linearity=non_linearity,
                jit_compile=False,
            )
        if policy is None:
            policy = NNPolicy.default(
                environment,
                layers=layers,
                non_linearity=non_linearity,
                log_scale=True,
                min_scale=-20.0,
                max_scale=1.5,
                jit_compile=False,
            )

        if fixed_temperature:
            eta = Constant(eta)
        else:
            eta = Learnable(eta, positive=True)

        policy_optimizer = Adam(policy.parameters(), lr=lr)
        critic_optimizer = Adam(critic.parameters(), lr=lr)
        temp_optimizer = Adam(eta.parameters(), lr=lr)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            eta=eta,
            actor_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            *args,
            **kwargs,
        )
