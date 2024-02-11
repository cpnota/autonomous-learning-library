from .a2c import A2CAtariPreset, a2c
from .c51 import C51AtariPreset, c51
from .ddqn import DDQNAtariPreset, ddqn
from .dqn import DQNAtariPreset, dqn
from .ppo import PPOAtariPreset, ppo
from .rainbow import RainbowAtariPreset, rainbow
from .vac import VACAtariPreset, vac
from .vpg import VPGAtariPreset, vpg
from .vqn import VQNAtariPreset, vqn
from .vsarsa import VSarsaAtariPreset, vsarsa

__all__ = [
    "a2c",
    "c51",
    "ddqn",
    "dqn",
    "ppo",
    "rainbow",
    "vac",
    "vpg",
    "vqn",
    "vsarsa",
]
