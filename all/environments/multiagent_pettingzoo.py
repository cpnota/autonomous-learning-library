import importlib
import numpy as np
import torch
import cloudpickle
import gym
from all.core import MultiagentState
from ._multiagent_environment import MultiagentEnvironment


class MultiagentPettingZooEnv(MultiagentEnvironment):
    '''
    A wrapper for generael PettingZoo environments (see: https://www.pettingzoo.ml/).

    This wrapper converts the output of the PettingZoo environment to PyTorch tensors,
    and wraps them in a State object that can be passed to an Agent.

    Args:
        zoo_env (AECEnv): A PettingZoo AECEnv environment (e.g. pettingzoo.mpe.simple_push_v2)
        device (optional): the device on which tensors will be stored
    '''

    def __init__(self, zoo_env, name, device='cuda'):
        env = zoo_env
        env.reset()
        self._env = env
        self._name = name
        self._device = device
        self.agents = self._env.agents
        self.subenvs = {
            agent: SubEnv(agent, device, self.state_spaces[agent], self.action_spaces[agent])
            for agent in self.agents
        }

    '''
    Reset the environment and return a new initial state.

    Returns:
        An initial MultiagentState object.
    '''

    def reset(self):
        self._env.reset()
        return self.last()

    '''
    Reset the environment and return a new initial state.

    Args:
        action (int): An int or tensor containing a single integer representing the action.

    Returns:
        The MultiagentState object for the next agent
    '''

    def step(self, action):
        if action is None:
            self._env.step(action)
            return
        self._env.step(self._convert(action))
        return self.last()

    def seed(self, seed):
        self._env.seed(seed)

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

    def agent_iter(self):
        return self._env.agent_iter()

    def is_done(self, agent):
        return self._env.dones[agent]

    def duplicate(self, n):
        return [MultiagentPettingZooEnv(cloudpickle.loads(cloudpickle.dumps(self._env)), self._name, device=self.device) for _ in range(n)]

    def last(self):
        observation, reward, done, info = self._env.last()
        selected_obs_space = self._env.observation_spaces[self._env.agent_selection]
        return MultiagentState.from_zoo(self._env.agent_selection, (observation, reward, done, info), device=self._device, dtype=selected_obs_space.dtype)

    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device

    @property
    def agent_selection(self):
        return self._env.agent_selection

    @property
    def state_spaces(self):
        return self._env.observation_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

    def _convert(self, action):
        agent = self._env.agent_selection
        action_space = self._env.action_spaces[agent]
        if torch.is_tensor(action):
            if isinstance(action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action


class SubEnv():
    def __init__(self, name, device, state_space, action_space):
        self.name = name
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

    @property
    def observation_space(self):
        return self.state_space
