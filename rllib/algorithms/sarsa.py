"""SARSA Algorithm."""
import copy

import torch

from rllib.util.neural_networks import update_parameters

from .abstract_algorithm import AbstractAlgorithm, TDLoss


class SARSA(AbstractAlgorithm):
    r"""Implementation of SARSA algorithm.

    SARSA is an on-policy model-free control algorithm.

    The SARSA algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma Q(s', a')
    where a' is sampled from a greedy policy w.r.t the current Q-Value estimate.

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma Q(s', a')
    .. math:: \mathcal{L}(Q(s, a), Q_{target})

    Parameters
    ----------
    q_function: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Rummery, G. A., & Niranjan, M. (1994).
    On-line Q-learning using connectionist systems. Cambridge, UK.

    Sutton, R. S. (1996).
    Generalization in reinforcement learning: Successful examples using sparse coarse
    coding. NIPS.

    Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000).
    Convergence results for single-step on-policy reinforcement-learning algorithms.
    Machine learning
    """

    def __init__(self, q_function, criterion, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion
        self.gamma = gamma

    def forward(self, trajectories):
        """Compute the loss and the td-error."""
        trajectory = trajectories[0]
        state, action = trajectory.state, trajectory.action
        reward, done = trajectory.reward, trajectory.done
        next_state, next_action = trajectory.next_state, trajectory.next_action
        pred_q = self.q_function(state, action)

        with torch.no_grad():
            next_v = self.q_target(next_state, next_action)
            target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return TDLoss(
            loss=self.criterion(pred_q, target_q), td_error=(pred_q - target_q).detach()
        )

    def update(self):
        """Update the target network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class GradientSARSA(SARSA):
    r"""Implementation of Gradient SARSA.

    The gradient SARSA algorithm takes the gradient of the target value too.

    .. math:: Q_{target} = (r(s, a) + \gamma Q(s', a')).detach()

    References
    ----------
    TODO: find
    """

    def forward(self, trajectories):
        """Compute the loss and the td-error."""
        trajectory = trajectories[0]
        state, action = trajectory.state, trajectory.action
        reward, done = trajectory.reward, trajectory.done
        next_state, next_action = trajectory.next_state, trajectory.next_action

        pred_q = self.q_function(state, action)

        next_v = self.q_function(next_state, next_action)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)
