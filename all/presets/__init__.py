from all.presets import atari
from all.presets import classic_control
from all.presets import continuous
from .preset import Preset
from .independent_multiagent import IndependentMultiagentPreset

__all__ = [
    "Preset",
    "atari",
    "classic_control",
    "continuous",
    "IndependentMultiagentPreset"
]
