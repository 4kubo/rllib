from rllib.dataset.datatypes import Action, State
from rllib.model import AbstractModel

from .abstract_system import AbstractSystem


class ModelSystem(AbstractSystem):
    dynamical_model: AbstractModel

    def __init__(self, dynamical_model: AbstractModel) -> None: ...

    def step(self, action: Action) -> State: ...

    def reset(self, state: State) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...
