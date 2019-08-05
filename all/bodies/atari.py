from ._body import Body
from .rewards import ClipRewards
from .vision import FrameStack

class DeepmindAtariBody(Body):
    def __init__(self, agent):
        agent = FrameStack(agent)
        agent = ClipRewards(agent)
        super().__init__(agent)
