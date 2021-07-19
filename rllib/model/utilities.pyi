from typing import Any, Iterable, List, Optional

from .abstract_model import AbstractModel


class PredictionStrategy(object):
    models: Iterable[AbstractModel]
    prediction_strategy: str
    _prediction_strategies: List[str]
    _heads: List[int]
    _sample_shape: List[int]
    def __init__(
            self,
            *models: AbstractModel,
            prediction_strategy: str = ...,
            sample_shape: Optional[List[int]] = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
