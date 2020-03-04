from rllib.value_function import NNEnsembleQFunction
from rllib.dataset.datatypes import State, Action, Reward, Done
from .abstract_dpg_agent import AbstractDPGAgent
from typing import Tuple, Any
from torch import Tensor


class TD3Agent(AbstractDPGAgent):
    q_function: NNEnsembleQFunction

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...