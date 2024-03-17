from .abstract import TargetNetwork
from .fixed import FixedTarget
from .polyak import PolyakTarget
from .trivial import TrivialTarget

__all__ = ["TargetNetwork", "FixedTarget", "PolyakTarget", "TrivialTarget"]
