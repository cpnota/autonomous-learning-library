import importlib
import numpy as np
import torch
import gym
from all.core import MultiAgentState
from ._multiagent_environment import MultiagentEnvironment


class MultiagentAtariEnv():
    '''
    A wrapper for PettingZoo Atari environments (see: https://www.pettingzoo.ml/atari).

    This wrapper converts the output of the PettingZoo environment to PyTorch tensors,
    and wraps them in a State object that can be passed to an Agent.

    Args:
        env_name (string): A string representing the name of the environment (e.g. pong-v1)
        device (optional): the device on which tensors will be stored
    '''
    def __init__(self, env_name, device='cuda'):
        env = self._load_env(env_name)
        # env = expand_dims(env, 0)
        env.reset()
        self._env = env
        self.name = env_name
        self.device = device
        self.agents = self._env.agents
        self.subenvs = {
            agent : SubEnv(agent, device, self.state_spaces[agent], self.action_spaces[agent])
            for agent in self.agents
        }

    '''
    Reset the environment and return a new intial state.

    Returns:
        An initial MultiagentState object.
    '''
    def reset(self):
        self._env.reset()
        return self.last()

    '''
    Reset the environment and return a new intial state.

    Args:
        action (int): An int or tensor containing a single integer representing the action.

    Returns:
        The MultiagentState object for the next agent
    '''
    def step(self, action):
        if action is None:
            self._env.step(action)
            return
        if torch.is_tensor(action):
            self._env.step(action.item())
        else:
            self._env.step(action)
        return self.last()

    def agent_iter(self):
        return self._env.agent_iter()

    @property
    def state_spaces(self):
        return { agent: gym.spaces.Box(0, 255, (1, 84, 84), np.uint8) for agent in self._env.possible_agents}

    @property
    def observation_spaces(self):
        return self.state_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def is_done(self, agent):
        return self._env.dones[agent]

    def last(self):
        observation, reward, done, info = self._env.last()
        observation = np.expand_dims(observation, 0)
        return MultiAgentState.from_zoo(self._env.agent_selection, (observation, reward, done, info), device='cuda', dtype=np.uint8)

    def _load_env(self, env_name):
        from pettingzoo import atari
        from supersuit import resize_v0, frame_skip_v0
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
        env = frame_skip_v0(env, 4)
        env = resize_v0(env, 84, 84)
        return env

class SubEnv():
    def __init__(self, name, device, state_space, action_space):
        self.name = name
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

    @property
    def observation_space(self):
        return self.state_space
