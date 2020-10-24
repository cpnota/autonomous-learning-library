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
        self.device = device
        self.agents = self._env.agents

    def reset(self):
        observation = self._env.reset()
        state = State.from_gym((observation.reshape((1, 84, 84),)), device=self.device, dtype=np.uint8)
        return state

    def step(self, action):
        observation = self._env.step(action.item())
        reward, done, info = self._env.last()
        return State.from_gym((observation.reshape((1, 84, 84)), reward, done, info), device='cuda', dtype=np.uint8)

    def agent_iter(self):
        return self._env.agent_iter()
