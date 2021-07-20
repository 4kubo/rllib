"""Vectorized Gym Environments."""
from gym.envs.registration import register

from .acrobot import DiscreteVectorizedAcrobotEnv, VectorizedAcrobotEnv
from .cartpole import DiscreteVectorizedCartPoleEnv, VectorizedCartPoleEnv
from .pendulum import VectorizedPendulumEnv

register(
    id="VecContinuous-CartPole-v0",
    entry_point="rllib.environment.vectorized.cartpole:VectorizedCartPoleEnv",
)

register(
    id="VecDiscrete-CartPole-v0",
    entry_point="rllib.environment.vectorized.cartpole:DiscreteVectorizedCartPoleEnv",
)

register(
    id="VecContinuous-Acrobot-v0",
    entry_point="rllib.environment.vectorized.acrobot:VectorizedAcrobotEnv",
)

register(
    id="VecDiscrete-Acrobot-v0",
    entry_point="rllib.environment.vectorized.acrobot:DiscreteVectorizedAcrobotEnv",
)

register(
    id="VecPendulum-v0",
    entry_point="rllib.environment.vectorized.pendulum:VectorizedPendulumEnv",
)
