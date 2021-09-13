"""Model implemented by querying an environment."""
import numpy as np
import torch

from rllib.environment.gym_environment import GymEnvironment
from rllib.environment.vectorized.subproc_vec_env import MujocoVecEnv
from .abstract_model import AbstractModel

VEC2ENV_NAME = {
    "PendulumSwingUp-v0": "VecPendulum-v0",
    "VecContinuous-CartPole-v0": "VecContinuous-CartPole-v0",
    "VecDiscrete-CartPole-v0": "VecDiscrete-CartPole-v0",
    "VecContinuous-Acrobot-v0": "VecContinuous-Acrobot-v0",
    "VecDiscrete-Acrobot-v0": "VecDiscrete-Acrobot-v0",
    "LQR-v0": "VecLQR-v0",
    # MuJoCo envs.
    "MBHalfCheetah-v0": "MBHalfCheetah-v0",
    "MBHumanoid-v0": "MBHumanoid-v0",
    "MBAnt-v0": "MBAnt-v0",
    "MBSwimmer-v0": "MBSwimmer-v0",
    "MBCartPole-v0": "MBCartPole-v0",
    "MBHopper-v0": "MBHopper-v0",
    "MBInvertedPendulum-v0": "MBInvertedPendulum-v0",
    "MBPusher-v0": "MBPusher-v0",
    "MBReacher2d-v0": "MBReacher2d-v0",
    "MBReacher3d-v0": "MBReacher3d-v0",
    "MBWalker2d-v0": "MBWalker2d-v0",
}


class EnvironmentModel(AbstractModel):
    """Implementation of a Dynamical Model implemented by querying an environment.

    Parameters
    ----------
    environment: AbstractEnvironment

    """

    def __init__(
        self, environment, n_remotes=None, model_kind="dynamics", **env_config
    ):
        super().__init__(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            deterministic=True,
            model_kind=model_kind,
        )

        try:
            vector_env_name = VEC2ENV_NAME[environment.name]
        except KeyError:
            raise NotImplementedError(
                "{0} is not implemented yet as a vector environment"
                " for use inside TrueModel".format(environment.name)
            )

        env_config = env_config or {}

        if not vector_env_name.startswith("Vec"):
            assert isinstance(n_remotes, int), ValueError(
                "n_remotes = {0}. If you use MujocoVecEnv for {1}, "
                "set integer for n_remotes > 0.".format(n_remotes, vector_env_name)
            )
            self.environment = MujocoVecEnv(
                env_name=vector_env_name, n_remotes=n_remotes, **env_config
            )
        else:
            self.environment = GymEnvironment(vector_env_name, **env_config)

        self.environment.reset()

    @classmethod
    def default(cls, environment, n_remotes=None, **kwargs):
        """See AbstractModel.default()."""
        return cls(environment, n_remotes=n_remotes, **kwargs)

    def forward(self, state, action, _=None):
        """Get Next-State distribution."""
        self.environment.state = state
        next_state, reward, done, _ = self.environment.step(action)
        if self.model_kind == "dynamics":
            if isinstance(next_state, np.ndarray):
                next_state = torch.tensor(next_state, dtype=torch.get_default_dtype())
            return next_state, torch.zeros(1)
        elif self.model_kind == "rewards":
            return reward, torch.zeros(1)
        elif self.model_kind == "termination":
            return (
                torch.zeros(*done.shape, 2)
                .scatter_(
                    dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf")
                )
                .squeeze(-1)
            )
        else:
            raise NotImplementedError(f"{self.model_kind} not implemented")

    @classmethod
    def is_available(self, env_name) -> bool:
        """Return whether env_name is available for EnvironmentModel."""
        return env_name in VEC2ENV_NAME

    @property
    def name(self):
        """Get Model name."""
        return f"{self.environment.name} Model"
