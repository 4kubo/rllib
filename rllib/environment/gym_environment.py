"""Wrapper for OpenAI-Gym Environments."""

import gym
from gym.wrappers import TimeLimit, Monitor

from .abstract_environment import AbstractEnvironment
from .utilities import parse_space
from .vectorized.dummy_vec_env import DummyVecEnv
from .vectorized.util import VectorizedEnv


class GymEnvironment(AbstractEnvironment):
    """Wrapper for OpenAI-Gym Environments.

    Parameters
    ----------
    env_name: str
        environment name
    seed: int, optional
        random seed to initialize environment.

    """

    def __init__(
        self,
        env_name,
        seed=None,
        num_envs=1,
        max_episode_steps=0,
        log_dir=None,
        **kwargs
    ):
        episodic = kwargs.pop("episodic", False)

        def make_env(idx):
            env = gym.make(env_name, **kwargs)
            env = _wrap_env(env, episodic, max_episode_steps, log_dir, idx)
            return env

        if 1 < num_envs:
            env = DummyVecEnv([make_env for _ in range(num_envs)])
            dim_action, num_actions = parse_space(env.single_action_space)
            dim_state, num_states = parse_space(env.single_observation_space)
        else:
            env = make_env(0)
            dim_action, num_actions = parse_space(env.action_space)
            dim_state, num_states = parse_space(env.observation_space)

        self.env = env
        self.env.seed(seed)
        self.env_name = env_name
        self.kwargs = kwargs

        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0
        self.metadata = self.env.metadata

    def add_wrapper(self, wrapper, **kwargs):
        """Add a wrapper for the environment."""
        self.env = wrapper(self.env, **kwargs)

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    def pop_wrapper(self):
        """Pop last wrapper."""
        self.env = self.env.env

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        next_state, reward, done, info = self.env.step(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        self._time += 1
        return next_state, reward, done, info

    def render(self, mode="human"):
        """See `AbstractEnvironment.render'."""
        return self.env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self.env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self.env.reset()

    @property
    def goal(self):
        """Return current goal of environment."""
        if hasattr(self.env, "goal"):
            return self.env.goal
        return None

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        if hasattr(self.env, "_get_obs"):
            return getattr(self.env, "_get_obs")()
        elif hasattr(self.env, "state"):
            return self.env.state
        elif hasattr(self.env, "s"):
            return self.env.s
        else:
            raise NotImplementedError("Strange state")

    @state.setter
    def state(self, value):
        if hasattr(self.env, "set_state"):
            if isinstance(self.env, VectorizedEnv):
                self.env.set_state(value)
            else:
                self.env.set_state(
                    value[: len(self.env.sim.data.qpos)],
                    value[len(self.env.sim.data.qpos) :],
                )
        elif hasattr(self.env, "state"):
            self.env.state = value
        elif hasattr(self.env, "s"):
            self.env.s = value
        else:
            raise NotImplementedError("Strange state")

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.env_name


def _wrap_env(env, episodic: int, max_episode_steps: int, log_dir=None, idx=0):
    # Time limit is controlled not when gym registration but by argument
    if isinstance(env, TimeLimit) and (not episodic):
        env = env.unwrapped
    if not isinstance(env, TimeLimit) and 0 < max_episode_steps:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # Wrap with montior wrapper
    if log_dir is not None and isinstance(log_dir, str):
        # TODO: Deal with issues when env is mujoco
        env = Monitor(env, directory=log_dir, uid=idx)
    return env
