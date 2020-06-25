import numpy as np
import torch
import numbers

class State(dict):
    def __init__(self, x, device='cpu'):
        if not 'observation' in x:
            raise Exception('State must contain an observation')
        if not 'reward' in x:
            x['reward'] = 0.
        if not 'done' in x:
            x['done'] = False
        if not 'mask' in x:
            x['mask'] = 1. - x['done']
        super().__init__(x)
        self.device = device

    @classmethod
    def from_list(cls, states):
        d = {}
        for key in states[0].keys():
            d[key] = [state[key] for state in states]
        return State(d, device=states[0].device)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__({k:v[key] for (k, v) in self.items()}, device=self.device)
        if isinstance(key, int):
            return self.__class__({k:v[key:key+1] for (k, v) in self.items()}, device=self.device)
        if torch.is_tensor(key):
            # some things may get los
            d = {}
            for (k, v) in self.items():
                try:
                    d[k] = v[key]
                except:
                    pass
            return self.__class__(d, device=self.device)
        try:
            value = super().__getitem__(key)
        except KeyError:
            return None
        if torch.is_tensor(value):
            return value
        if isinstance(value, list):
            try:
                if torch.is_tensor(value[0]):
                    return torch.cat(value)
                if isinstance(value[0], numbers.Number):
                    return torch.tensor(value, device=self.device)
            except:
                return value
        return value

    def flatten(self):
        for k in self.keys():
            self[k] = self[k]
        return self

    def update(self, key, value):
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        return self.__class__(x, device=self.device)

    @classmethod
    def from_gym(cls, state, device='cpu', dtype=np.float32):
        if not isinstance(state, tuple):
            return State({
                'observation': torch.from_numpy(
                    np.array(
                        state,
                        dtype=dtype
                    ),
                ).unsqueeze(0).to(device)
            }, device=device)

        observation, reward, done, info = state
        observation = torch.from_numpy(
            np.array(
                observation,
                dtype=dtype
            ),
        ).unsqueeze(0).to(device)
        # mask = DONE.to(device) if done else NOT_DONE.to(device)
        x = {
            'observation': observation,
            'reward': float(reward),
            'done': done,
        }
        info = info if info else {}
        for key in info:
            x[key] = info[key]
        return State(x, device=device)

    @property
    def observation(self):
        return self['observation']

    @property
    def reward(self):
        return self['reward']

    @property
    def done(self):
        return self['done']

    @property
    def mask(self):
        return self['mask']

    def __len__(self):
        return len(self.observation)

DONE = torch.tensor(
    [0],
    dtype=torch.uint8,
)

NOT_DONE = torch.tensor(
    [1],
    dtype=torch.uint8,
)
