from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch.nn as nn
from numpy import ndarray
from torch import Tensor, device
from torch.utils import data

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import AbstractTransform

T = TypeVar("T", bound="ExperienceReplay")

class ExperienceReplay(data.Dataset):
    max_len: int
    memory: ndarray
    valid: Tensor
    weights: Tensor
    transformations: List[AbstractTransform]
    data_count: int
    _num_steps: int
    zero_observation: Optional[Observation]
    raw: bool
    def __init__(
        self,
        max_len: int,
        transformations: Optional[Union[List[AbstractTransform], nn.ModuleList]] = ...,
        num_steps: int = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, num_steps: Optional[int] = ...) -> T: ...
    def split(self, ratio: float = ..., *args: Any, **kwargs: Any) -> Tuple[T, T]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, item: int) -> Tuple[Dict[str, Tensor], int, Tensor]: ...

    def _init_observation(self, observation: Observation) -> None: ...

    def _get_consecutive_observations(
            self, start_idx: int, num_steps: int
    ) -> Observation: ...

    def _get_observation(self, idx: int) -> Observation: ...

    def reset(self) -> None: ...

    def end_episode(self) -> None: ...

    def append(self, observation: Observation) -> None: ...

    def append_invalid(self) -> None: ...

    def sample_batch(self, batch_size: int, device: device) -> Tuple[
        Observation, Tensor, Tensor]: ...

    def update(self, indexes: Tensor, td_error: Tensor) -> None: ...

    @property
    def all_data(self) -> Observation: ...

    @property
    def all_raw(self) -> Observation: ...

    @property
    def is_full(self) -> bool: ...

    @property
    def ptr(self) -> int: ...
    @property
    def valid_indexes(self) -> Tensor: ...
    @property
    def num_steps(self) -> int: ...
    @num_steps.setter
    def num_steps(self, value: int) -> None: ...
