"""Implementation of REINFORCE Algorithms."""

from rllib.agent.on_policy_agent import OnPolicyAgent
from rllib.algorithms.reinforce import REINFORCE
from rllib.dataset.utilities import stack_list_of_tuples


class REINFORCEAgent(OnPolicyAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, env_name, policy, policy_optimizer, baseline=None,
                 baseline_optimizer=None, criterion=None, num_rollouts=1, num_iter=1,
                 target_update_frequency=1, gamma=1.0, exploration_steps=0,
                 exploration_episodes=0):
        super().__init__(env_name,
                         num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.algorithm = REINFORCE(policy, baseline, criterion(reduction='none'),
                                   gamma)
        self.policy = self.algorithm.policy
        self.policy_optimizer = policy_optimizer
        self.baseline_optimizer = baseline_optimizer

        self.num_iter = num_iter
        self.target_update_frequency = target_update_frequency

    def _train(self):
        """See `AbstractAgent.train_agent'."""
        trajectories = [stack_list_of_tuples(t) for t in self.trajectories]

        for _ in range(self.num_iter):
            # Update actor.
            self.policy_optimizer.zero_grad()
            if self.baseline_optimizer is not None:
                self.baseline_optimizer.zero_grad()

            losses = self.algorithm(trajectories)
            losses.actor_loss.backward()
            self.policy_optimizer.step()

            # Update Logs.
            self.logger.update(actor_losses=losses.actor_loss.item())

            # Update baseline.
            if self.baseline_optimizer is not None:
                losses.baseline_loss.backward()
                self.baseline_optimizer.step()
                self.logger.update(baseline_losses=losses.baseline_loss.item())

            self.train_iter += 1
            if self.train_iter % self.target_update_frequency == 0:
                self.algorithm.update()
