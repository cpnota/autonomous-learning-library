import gym
import torch
from all.core import State
from ._vector_environment import VectorEnvironment
import numpy as np


class DuplicateEnvironment(VectorEnvironment):
    '''
    Turns a list of ALL Environment objects into a VectorEnvironment object

    This wrapper just takes the list of States the environments generate and outputs
    a StateArray object containing all of the environment states. Like all vector
    environments, the sub environments are automatically reset when done.

    Args:
        envs: A list of ALL environments
        device (optional): the device on which tensors will be stored
    '''

    def __init__(self, envs, device=torch.device('cpu')):
        self._name = envs[0].name
        self._envs = envs
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
        self._state = State.array([sub_env.reset() for sub_env in self._envs])
        return self._state

    def step(self, actions):
        states = []
        actions = actions.cpu().detach().numpy()
        for sub_env, action in zip(self._envs, actions):
            state = sub_env.reset() if sub_env.state.done else sub_env.step(action)
            states.append(state)
        self._state = State.array(states)
        return self._state

    def close(self):
        return self._env.close()

    def seed(self, seed):
        for i, env in enumerate(self._envs):
            env.seed(seed + i)

    @property
    def state_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

    @property
    def state_array(self):
        return self._state

    @property
    def device(self):
        return self._device

    @property
    def num_envs(self):
        return len(self._envs)
