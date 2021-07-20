"""Vectorized Gym MuJoCo environment which use ray package"""
from time import sleep

import gym
import numpy as np
import ray
import torch

from rllib.environment.gym_environment import GymEnvironment


@ray.remote
class RemoteMujocoEnv:
    """
    Remote environment class as a ray's actor
    """

    def __init__(self, env_creator):
        self._env = env_creator()

    def step(self, action, qpos, qvel):
        """
        Simulate one timestep of the environment according to each actions
         at the states of `qpos` and `qvel`
        """
        tmp_list = []
        for a, p, v in zip(action, qpos, qvel):
            self._env.set_state(p, v)
            tmp_list.append(self._env.step(a))
        return list(map(np.asarray, zip(*tmp_list)))

    def reset(self):
        return self._env.reset()

    def base_env(self):
        return self._env

    def seed(self, seed=0):
        self._env.seed(seed)


class MujocoVecEnv(GymEnvironment):
    """
    VecEnv that runs multiple MuJoCo environments in parallel in subprocesses
    and communicates with them via pipes.
    Recommended to use when n_remotes > 1 and step() can be a bottleneck.
    """

    _batch_shape = None
    _shared_seed: int = None
    _device = None
    _split_qpos = None
    _split_qvel = None

    def __init__(self, env_name, n_remotes, shared_seed=True, **env_configs):
        """

        Args:
            env_name: str
                Environment id registered in gym
            n_remotes: int
                The number environments as ray's remote class
            shared_seed: bool
                Flag about whether share seed over remote environments or not
            env_configs: dict
                config passed to gym make call
        """
        super().__init__(env_name, **env_configs)
        self.n_envs = n_remotes
        self._shared_seed = shared_seed
        self._pending = {}
        self._dtype = torch.get_default_dtype()

        if not ray.is_initialized():
            ray.init()

        def env_creator(**env_config):
            remote_env = RemoteMujocoEnv.remote(
                lambda: gym.make(env_name, **env_config)
            )
            sleep(2)
            return remote_env

        self._envs = [env_creator(**env_configs) for _ in range(n_remotes)]
        self.seed(0)

        self.n_pos = len(self.env.sim.data.qpos)
        self.n_vel = len(self.env.sim.data.qvel)

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        return super().state

    @state.setter
    def state(self, value):
        qpos, qvel = value[..., : self.n_pos], value[..., self.n_pos :]
        self._batch_shape = qpos.shape[:-1]
        self._split_qpos = np.array_split(
            qpos.detach().cpu().numpy().reshape(-1, self.n_pos), self.n_envs, axis=0
        )
        self._split_qvel = np.array_split(
            qvel.detach().cpu().numpy().reshape(-1, self.n_vel), self.n_envs, axis=0
        )

    def step_async(self, actions):
        """Send actions and states to remote environments."""
        assert actions.shape[:-1] == self._batch_shape
        self._device = actions.device

        actions = actions.reshape(-1, actions.shape[-1]).detach().cpu().numpy()
        actions = np.array_split(actions, self.n_envs, axis=0)

        for env, action, qpos, qvel in zip(
            self._envs, actions, self._split_qpos, self._split_qvel
        ):
            obs_refs = env.step.remote(action, qpos, qvel)
            self._pending[obs_refs] = env

    def step_wait(self):
        """Get results from remote environments of applying actions."""
        not_ready = [None]
        ready = []

        while not_ready:
            ready, not_ready = ray.wait(
                list(self._pending),
                num_returns=self.n_envs,
                timeout=0.5,
            )
            # sleep(0.01)

        tmp_list = []
        for obj_ref in ready:
            self._pending.pop(obj_ref)
            ob = ray.get(obj_ref)
            tmp_list.append(ob)
        return_list = [list(ret) for ret in zip(*tmp_list)]
        observations = torch.tensor(
            np.vstack(return_list[0]),
            device=self._device,
            dtype=self._dtype,
        )
        rewards = torch.tensor(
            np.hstack(return_list[1]), device=self._device, dtype=self._dtype
        )
        dones = torch.tensor(
            np.hstack(return_list[2]), device=self._device, dtype=torch.bool
        )
        return observations, rewards, dones, None

    def step(self, actions):
        """See `AbstractEnvironment.step'."""
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed: int = 0):
        """Set seeds for each remote environment."""
        for env in self._envs:
            env.seed.remote(seed)
            if not self._shared_seed:
                seed += 1

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        for env in self._envs:
            env.reset.remote()
