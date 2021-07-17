from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch.__spec__ as torch_mod
from torch import Tensor, device
from torch.distributions import Distribution

from rllib.dataset.datatypes import Array, Reward, TupleDistribution


def get_backend(array: Array) -> Union[np, torch_mod]: ...
def set_random_seed(seed: int) -> None: ...
def save_random_state(directory: str) -> None: ...
def load_random_state(directory: str) -> None: ...
def mellow_max(values: Array, omega: Union[Tensor, float] = ...) -> Array: ...
def integrate(
    function: Callable,
    distribution: Distribution,
    out_dim: Optional[int] = ...,
    num_samples: int = ...,
) -> Tensor: ...
def tensor_to_distribution(args: TupleDistribution, **kwargs: Any) -> Distribution: ...
def separated_kl(
    p: Distribution, q: Distribution, log_p: Tensor = ..., log_q: Tensor = ...
) -> Tuple[Tensor, Tensor]: ...
def off_policy_weight(
        eval_log_p: Tensor,
        behavior_log_p: Tensor,
        full_trajectory: bool = ...,
        clamp_max: float = ...,
) -> Tensor: ...


def get_entropy_and_log_p(
        pi: Distribution, action: Tensor, action_scale: Union[float, Tensor]
) -> Tuple[Tensor, Tensor]: ...


def sample_mean_and_cov(sample: Tensor, diag: bool = ..., device: device = ...,
                        ) -> Tuple[Tensor, Tensor]: ...


def safe_cholesky(covariance_matrix: Tensor, jitter: float = ...) -> Tensor: ...


class MovingAverage(object):
    _count: int
    _total_value: float

    def __init__(self) -> None: ...

    def update(self, value: float) -> None: ...

    @property
    def value(self) -> float: ...

def moving_average_filter(x: Array, y: Array, horizon: int) -> Array: ...

class RewardTransformer(object):
    offset: float
    low: float
    high: float
    scale: float
    def __init__(
        self,
        offset: float = ...,
        low: float = ...,
        high: float = ...,
        scale: float = ...,
    ) -> None: ...
    def __call__(self, reward: Reward) -> Reward: ...

class TimeIt(object):
    name: str
    start: float
    def __init__(self, name: str = ...) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
