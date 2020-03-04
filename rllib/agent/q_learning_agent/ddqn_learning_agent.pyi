from .abstract_q_learning_agent import AbstractQLearningAgent
from rllib.dataset.datatypes import State, Action, Reward, Done
from torch import Tensor
from typing import Tuple, Any


class DDQNAgent(AbstractQLearningAgent):

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...
