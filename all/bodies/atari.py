from all.environments import State
from ._body import Body
from .rewards import ClipRewards
from .vision import FrameStack

class DeepmindAtariBody(Body):
    def __init__(self, agent, lazy_frames=False, episodic_lives=True, frame_stack=4):
        agent = FrameStack(agent, lazy=lazy_frames, size=frame_stack)
        agent = ClipRewards(agent)
        if episodic_lives:
            agent = EpisodicLives(agent)
        super().__init__(agent)

class EpisodicLives(Body):
    def act(self, state, reward):

        return self.agent.act(self._done_on_life_lost(state), reward)

    def eval(self, state, reward):
        return self.agent.eval(self._done_on_life_lost(state), reward)

    def _done_on_life_lost(self, state):
        for i in range(len(state)):
            if state.info[i]['life_lost']:
                mask = state.mask.clone()
                mask[i] = 0
                state = State(state.raw, mask=mask, info=state.info)
        return state
