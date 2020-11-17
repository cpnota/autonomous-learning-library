from .a2c import a2c, A2CAtariPreset
from .c51 import c51, C51AtariPreset
from .dqn import dqn, DQNAtariPreset
from .ddqn import ddqn, DDQNAtariPreset
from .ppo import ppo, PPOAtariPreset
from .rainbow import rainbow, RainbowAtariPreset
from .vac import vac, VACAtariPreset
from .vpg import vpg, VPGAtariPreset
from .vqn import vqn, VQNAtariPreset
from .vsarsa import vsarsa, VSarsaAtariPreset
from .preset import Preset


__all__ = [
    "a2c",
    "A2CAtariPreset",
    "c51",
    "C51AtariPreset",
    "ddqn",
    "DDQNAtariPreset",
    "dqn",
    "DQNAtariPreset",
    "ppo",
    "PPOAtariPreset",
    "rainbow",
    "RainbowAtariPreset",
    "vac",
    "VACAtariPreset",
    "vpg",
    "VPGAtariPreset",
    "vqn",
    "VQNAtariPreset",
    "vsarsa",
    "VSarsaAtariPreset",
    "Preset"
]
