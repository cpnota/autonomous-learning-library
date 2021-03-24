import gym
import torch
from all.core import StateArray
from ._vector_environment import VectorEnvironment
import cloudpickle
import numpy as np


class GymVectorEnvironment(VectorEnvironment):
    '''
    A wrapper for Gym's vector environments
    (see: https://github.com/openai/gym/blob/master/gym/vector/vector_env.py).

    This wrapper converts the output of the vector environment to PyTorch tensors,
    and wraps them in a StateArray object that can be passed to a Parallel Agent.
    This constructor accepts a preconstructed gym vetor environment. Note that
    in the latter case, the name property is set to be the whatever the name
    of the outermost wrapper on the environment is.

    Args:
        vec_env: An OpenAI gym vector environment
        device (optional): the device on which tensors will be stored
    '''

    def __init__(self, vec_env, name, device=torch.device('cpu')):
        self._name = name
        self._env = vec_env
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

    @property
    def name(self):
        return self._name

    def reset(self):
        state_tuple = self._env.reset(), np.zeros(self._env.num_envs), np.zeros(self._env.num_envs), None
        self._state = self._to_state(*state_tuple)
        return self._state

    def _to_state(self, obs, rew, done, info):
        obs = obs.astype(self.observation_space.dtype)
        rew = rew.astype("float32")
        done = done.astype("bool")
        mask = (1 - done).astype("float32")
        return StateArray({
            "observation": torch.tensor(obs, device=self._device),
            "reward": torch.tensor(rew, device=self._device),
            "done": torch.tensor(done, device=self._device),
            "mask": torch.tensor(mask, device=self._device)
        }, shape=(self._env.num_envs,))

    def step(self, action):
        state_tuple = self._env.step(action.cpu().detach().numpy())
        self._state = self._to_state(*state_tuple)
        return self._state

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    @property
    def state_space(self):
        return getattr(self._env, "single_observation_space", getattr(self._env, "observation_space"))

    @property
    def action_space(self):
        return getattr(self._env, "single_action_space", getattr(self._env, "action_space"))

    @property
    def state_array(self):
        return self._state

    @property
    def device(self):
        return self._device

    @property
    def num_envs(self):
        return self._env.num_envs
