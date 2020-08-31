from typing import Any, Union

from rllib.algorithms.reps import REPS
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .off_policy_agent import OffPolicyAgent

class REPSAgent(OffPolicyAgent):
    algorithm: REPS
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        epsilon: Union[float, ParameterDecay] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def _optimizer_dual(self) -> None: ...
    def _fit_policy(self) -> None: ...
    def _optimize_loss(self, loss_name: str = ...) -> None: ...
