import numpy as np
import torch

class State:
    def __init__(self, raw, mask=None, info=None):
        self._raw = raw

        if mask is None:
            self._mask = torch.ones(
                len(raw),
                dtype=torch.uint8,
                device=raw.device
            )
        else:
            self._mask = mask

        self._info = info or [None] * len(raw)

    @classmethod
    def from_list(cls, states):
        raw = torch.cat([state.features for state in states])
        done = torch.cat([state.mask for state in states])
        info = sum([state.info for state in states], [])
        return cls(raw, done, info)

    @classmethod
    def from_gym(cls, numpy_arr, done, info, device='cpu', dtype=np.float32):
        raw = torch.from_numpy(
            np.array(
                numpy_arr,
                dtype=dtype
            )
        ).unsqueeze(0).to(device)
        mask = DONE.to(device) if done else NOT_DONE.to(device)
        return cls(raw, mask=mask, info=[info])

    @property
    def features(self):
        '''
        Default features are the raw state.
        Override this method for other types of features.
        '''
        return self._raw

    @property
    def mask(self):
        return self._mask

    @property
    def info(self):
        return self._info

    @property
    def raw(self):
        return self._raw

    @property
    def done(self):
        return not self._mask

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return State(
                self._raw[idx],
                self._mask[idx],
                self._info[idx]
            )
        if isinstance(idx, torch.Tensor):
            return State(
                self._raw[idx],
                self._mask[idx],
                # can't copy info
            )
        return State(
            self._raw[idx].unsqueeze(0),
            self._mask[idx].unsqueeze(0),
            [self._info[idx]]
        )

    def __len__(self):
        return len(self._raw)

DONE = torch.tensor(
    [0],
    dtype=torch.uint8,
)

NOT_DONE = torch.tensor(
    [1],
    dtype=torch.uint8,
)
