from .a2c import a2c, A2CClassicControlPreset
from .c51 import c51, C51ClassicControlPreset
from .ddqn import ddqn, DDQNClassicControlPreset
from .dqn import dqn, DQNClassicControlPreset
from .ppo import ppo, PPOClassicControlPreset
from .rainbow import rainbow, RainbowClassicControlPreset
from .vac import vac, VACClassicControlPreset
from .vpg import vpg, VPGClassicControlPreset
from .vqn import vqn, VQNClassicControlPreset
from .vsarsa import vsarsa, VSarsaClassicControlPreset

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
