from typing import Any

import torch.nn as nn

from rllib.dataset.datatypes import Array, Observation

from .abstract_transform import AbstractTransform

class Clipper(nn.Module):
    _min: float
    _max: float
    def __init__(self, min_val: float, max_val: float) -> None: ...
    def forward(self, *array: Array, **kwargs: Any) -> Array: ...
    def inverse(self, array: Array) -> Array: ...

class RewardClipper(AbstractTransform):
    _clipper: Clipper
    def __init__(self, min_reward: float = ..., max_reward: float = ...) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...

class ActionClipper(AbstractTransform):
    _clipper: Clipper
    def __init__(self, min_action: float = ..., max_action: float = ...) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...
