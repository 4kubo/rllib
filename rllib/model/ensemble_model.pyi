from typing import Any, Optional, List

import torch
import torch.jit
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from .nn_model import NNModel


class EnsembleModel(NNModel):
    num_heads: int
    nn: torch.nn.ModuleList
    def __init__(
        self, num_heads: int, prediction_strategy: str = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def sample_posterior(self) -> None: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
    @torch.jit.export
    def set_head(self, head_ptr: int) -> None: ...

    @torch.jit.export
    def get_head(self) -> int: ...

    @torch.jit.export
    def set_head_idx(self, head_ptr: Tensor) -> None: ...

    @torch.jit.export
    def get_head_idx(self) -> Tensor: ...

    @torch.jit.export
    def set_prediction_strategy(
            self,
            prediction: str,
            shape: Optional[List[int]] = ...,
    ) -> None: ...

    @torch.jit.export
    def get_prediction_strategy(self) -> str: ...
