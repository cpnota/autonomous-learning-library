import torch
from ._body import Body
from .rewards import ClipRewards
from .vision import FrameStack


class DeepmindAtariBody(Body):
    def __init__(self, agent, lazy_frames=False, episodic_lives=True, frame_stack=4, clip_rewards=True):
        if frame_stack > 1:
            agent = FrameStack(agent, lazy=lazy_frames, size=frame_stack)
        if clip_rewards:
            agent = ClipRewards(agent)
        if episodic_lives:
            agent = EpisodicLives(agent)
        super().__init__(agent)


class EpisodicLives(Body):
    def process_state(self, state):
        if 'life_lost' not in state:
            return state

        if len(state.shape) == 0:
            if state['life_lost']:
                return state.update('mask', 0.)
            return state

        masks = [None] * len(state)
        life_lost = state['life_lost']
        for i, old_mask in enumerate(state.mask):
            if life_lost[i]:
                masks[i] = 0.
            else:
                masks[i] = old_mask
        return state.update('mask', torch.tensor(masks, device=state.device))
