import importlib
import numpy as np
import torch
from pettingzoo import atari
from supersuit import resize_v0, frame_skip_v0, frame_stack_v0, sticky_actions_v0, color_reduction_v0
from supersuit.gym_wrappers import BasicObservationWrapper
from all.core import MultiAgentState



class MultiAgentAtariEnv():
    def __init__(self, env_name, device='cuda'):
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
        env = frame_skip_v0(env, 4)
        env = resize_v0(env, 84, 84)
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

    def reset(self):
        self._env.reset()
        return self.last()

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
        return self._env.observation_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

    def render(self):
        return self._env.render(mode='human')

    def is_done(self, agent):
        return self._env.dones[agent]

    def last(self):
        observation, reward, done, info = self._env.last()
        observation = np.expand_dims(observation, 0)
        return MultiAgentState.from_zoo(self._env.agent_selection, (observation, reward, done, info), device='cuda', dtype=np.uint8)


class SubEnv():
    def __init__(self, name, device, state_space, action_space):
        self.name = name
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

    @property
    def observation_space(self):
        return self.state_space
