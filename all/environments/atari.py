from .gym_wrapper import GymWrapper
from .preprocessors import to_grayscale, downsample, to_torch

def make_atari(env):
    return GymWrapper(env, preprocessors=[downsample, to_grayscale, to_torch])
