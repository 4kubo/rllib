from typing import Tuple, List

import torch

from rllib.dataset.datatypes import Action, State, Reward, Done
from rllib.environment.gym_environment import GymEnvironment


class MujocoVecEnv(GymEnvironment):
    n_envs: int
    _pending: dict
    _dtype: torch.dtype
    _batch_shape: List[int]
    _shared_seed: int
    _device = torch.device
    _split_qpos = List[int]
    _split_qvel = List[int]
    _envs = List

    def __init__(
            self,
            env_name: str,
            n_remotes: int,
            shared_seed: bool = ...,
            **env_configs: dict
    ) -> None: ...

    def step_async(self, actions: Action) -> None: ...

    def step_wait(self) -> Tuple[State, Reward, Done, None]: ...

    def seed(self, seed: int) -> None: ...
