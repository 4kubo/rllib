"""Import environments."""
from gym.envs.registration import register

# from rllib.environment.systems.linear_system import LinearSystem


register(
    id="LQR-v0",
    entry_point="rllib.environment.system_environments.lqr_env:LQREnv",
    kwargs={"dim_state": 2, "dim_action": 2},
)

register(
    id="VecLQR-v0",
    entry_point="rllib.environment.system_environments.lqr_env:LQREnv",
    kwargs={"dim_state": 2, "dim_action": 2},
)
