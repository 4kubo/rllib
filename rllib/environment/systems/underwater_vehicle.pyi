from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem


class UnderwaterVehicle(ODESystem):

    def __init__(self, step_size: float = 0.01) -> None: ...

    def drag_force(self, velocity: State) -> State: ...

    def thrust(self, velocity: State, thrust: Action) -> State: ...

    def _ode(self, _: float, state: State, action: Action) -> State: ...
