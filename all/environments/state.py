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
        return StateList(states)

    def apply(self, model, *keys):
        return self.apply_mask(model(*[self.as_input(key) for key in keys])).squeeze(0)

    def as_input(self, key):
        return self[key].unsqueeze(0)

    def apply_mask(self, tensor):
        return tensor * self.mask

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
                ).to(device)
            }, device=device)

        observation, reward, done, info = state
        observation = torch.from_numpy(
            np.array(
                observation,
                dtype=dtype
            ),
        ).to(device)
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
        return 1

class StateList(State):
    def __init__(self, list_of_states, device=None):
        device = device if device else list_of_states[0].device
        x = {}
        self._len = len(list_of_states)
        for key in list_of_states[0].keys():
            v = list_of_states[0][key]
            try:
                if torch.is_tensor(v):
                    x[key] = torch.stack([state[key] for state in list_of_states])
                else:
                    x[key] = torch.tensor([state[key] for state in list_of_states], device=device)
            except:
                pass
        super().__init__(x, device=device)

    def update(self, key, value):
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        return self.__class__(x, device=self.device)

    def apply(self, model, *keys):
        return self.apply_mask(model(*[self.as_input(key) for key in keys])).squeeze(0)

    def as_input(self, key):
        return self[key]

    def apply_mask(self, tensor):
        return tensor * self['mask'].unsqueeze(-1)

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

    def __getitem__(self, key):
        # if isinstance(key, slice):
        #     return self.__class__({k:v[key] for (k, v) in self.items()}, device=self.device)
        # if isinstance(key, int):
        #     return self.__class__({k:v[key] for (k, v) in self.items()}, device=self.device)
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
        return value

    def __len__(self):
        return self._len
