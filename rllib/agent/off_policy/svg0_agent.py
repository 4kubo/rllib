"""Implementation of SVG-0 Algorithm."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.svg0 import SVG0
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class SVG0Agent(OffPolicyAgent):
    """Implementation of the SVG-0 Agent.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: AbstractPolicy
        policy that is learned.
    criterion: nn.Module.
        criterion of critic.

    References
    ----------
    Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T., & Tassa, Y. (2015).
    Learning continuous control policies by stochastic value gradients. NeuRIPS.

    """

    def __init__(self, critic, policy, criterion=loss.MSELoss, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert not policy.deterministic, "Policy must be stochastic."
        self.algorithm = SVG0(
            critic=critic,
            policy=policy,
            criterion=criterion(reduction="none"),
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        critic = NNQFunction.default(environment)
        policy = NNPolicy.default(environment)

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            optimizer=optimizer,
            policy_update_frequency=2,
            clip_gradient_val=10,
            *args,
            **kwargs,
        )
