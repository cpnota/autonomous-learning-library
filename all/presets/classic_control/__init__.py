from .a2c import A2CClassicControlPreset, a2c
from .c51 import C51ClassicControlPreset, c51
from .ddqn import DDQNClassicControlPreset, ddqn
from .dqn import DQNClassicControlPreset, dqn
from .ppo import PPOClassicControlPreset, ppo
from .rainbow import RainbowClassicControlPreset, rainbow
from .vac import VACClassicControlPreset, vac
from .vpg import VPGClassicControlPreset, vpg
from .vqn import VQNClassicControlPreset, vqn
from .vsarsa import VSarsaClassicControlPreset, vsarsa

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
