from ._body import Body
from .rewards import ClipRewards
from .vision import FrameStack

class DeepmindAtariBody(Body):
    def __init__(self, agent, lazy_frames=False):
        agent = FrameStack(agent, lazy=lazy_frames)
        agent = ClipRewards(agent)
        super().__init__(agent)
