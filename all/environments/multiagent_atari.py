import importlib
import numpy as np
import torch
from pettingzoo import atari
from supersuit import resize_v0, frame_skip_v0, frame_stack_v0, sticky_actions_v0, color_reduction_v0
from all.core import State


class MultiAgentAtariEnv():
    def __init__(self, env_name, device='cuda'):
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
        env = frame_skip_v0(env, 4)
        env = resize_v0(env, 84, 84)
        self._env = env
        self.name = env_name
        self.device = device
        self.agents = self._env.agents
        self.subenvs = {
            agent : SubEnv(agent, device, self.state_spaces[agent], self.action_spaces[agent])
            for agent in self.agents
        }

    def reset(self):
        observation = self._env.reset()
        state = State.from_gym((observation.reshape((1, 84, 84),)), device=self.device, dtype=np.uint8)
        return state

    def step(self, action):
        if torch.is_tensor(action):
            observation = self._env.step(action.item())
        else:
            observation = self._env.step(action)
        reward, done, info = self._env.last()
        return State.from_gym((observation.reshape((1, 84, 84)), reward, done, info), device='cuda', dtype=np.uint8)

    def agent_iter(self):
        return self._env.agent_iter()

    @property
    def state_spaces(self):
        return self._env.observation_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

class SubEnv():
    def __init__(self, name, device, state_space, action_space):
        self.name = name
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

    @property
    def observation_space(self):
        return self.state_space
