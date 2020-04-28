from abc import ABCMeta
from typing import Dict, List

from rllib.dataset.datatypes import Observation, State, Action, Distribution
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger


class AbstractAgent(object, metaclass=ABCMeta):
    policy: AbstractPolicy
    environment: str
    pi: Distribution
    counters: Dict[str, int]
    episode_steps: List[int]
    logger: Logger
    gamma: float
    exploration_steps: int
    exploration_episodes: int
    _training: bool
    comment: str

    last_trajectory: List[Observation]

    def __init__(self, environment: str, gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0, comment: str = '') -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def end_interaction(self) -> None: ...

    def _train(self) -> None: ...

    def train(self, val: bool = True) -> None: ...

    def eval(self, val: bool = True) -> None: ...

    @property
    def total_episodes(self) -> int: ...

    @property
    def total_steps(self) -> int: ...

    @property
    def name(self) -> str: ...
