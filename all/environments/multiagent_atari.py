import importlib
import numpy as np
import torch
import gym
from all.core import MultiagentState
from ._multiagent_environment import MultiagentEnvironment
from .multiagent_pettingzoo import MultiagentPettingZooEnv


class MultiagentAtariEnv(MultiagentPettingZooEnv):
    '''
    A wrapper for PettingZoo Atari environments (see: https://www.pettingzoo.ml/atari).

    This wrapper converts the output of the PettingZoo environment to PyTorch tensors,
    and wraps them in a State object that can be passed to an Agent.

    Args:
        env_name (string): A string representing the name of the environment (e.g. pong-v1)
        device (optional): the device on which tensors will be stored
    '''

    def __init__(self, env_name, device='cuda', **pettingzoo_params):
        env = self._load_env(env_name, pettingzoo_params)
        super().__init__(env, name=env_name, device=device)

    def _load_env(self, env_name, pettingzoo_params):
        from pettingzoo import atari
        from supersuit import resize_v0, frame_skip_v0, reshape_v0, max_observation_v0
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image', **pettingzoo_params)
        env = max_observation_v0(env, 2)
        env = frame_skip_v0(env, 4)
        env = resize_v0(env, 84, 84)
        env = reshape_v0(env, (1, 84, 84))
        return env
