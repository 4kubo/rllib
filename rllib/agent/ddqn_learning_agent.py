"""Implementation of DQNAgent Algorithms."""
from rllib.agent.q_learning_agent import QLearningAgent
from rllib.algorithms.q_learning import DDQN


class DDQNAgent(QLearningAgent):
    """Implementation of a DQN-Learning agent.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    optimizer: nn.optim
        Optimization algorithm for q_function.
    memory: ExperienceReplay
        Memory where to store the observations.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Hasselt, H. V. (2010).
    Double Q-learning. NIPS.

    Van Hasselt, Hado, Arthur Guez, and David Silver. (2016)
    Deep reinforcement learning with double q-learning. AAAI.
    """

    def __init__(self, env_name, q_function, policy, criterion, optimizer,
                 memory, num_iter=1, batch_size=64, target_update_frequency=4,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(env_name, q_function, policy, criterion, optimizer, memory,
                         num_iter, batch_size, target_update_frequency, gamma,
                         exploration_steps, exploration_episodes)
        self.algorithm = DDQN(q_function, criterion(reduction='none'), self.gamma)
