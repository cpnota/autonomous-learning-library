import torch

class State:
    def __init__(self, raw, done=None, info=None):
        self._raw = raw
        self._done = done
        if done is None:
            self._done = torch.ones(
                len(raw),
                dtype=torch.uint8,
                device=raw.device
            )
        self._info = info

    @classmethod
    def from_list(cls, states):
        raw = torch.cat([state.raw for state in states])
        done = torch.cat([state.done for state in states])
        info = [state.info for state in states]
        return cls(raw, done, info)

    @property
    def features(self):
        '''
        Default features are the raw state.
        Override this method for other types of features.
        '''
        return self._raw

    @property
    def done(self):
        return self._done

    @property
    def mask(self):
        return self._done

    @property
    def info(self):
        return self._info

    @property
    def raw(self):
        return self._raw

    def __getitem__(self, idx):
        return State(
            self._raw[idx].unsqueeze(0),
            self._done[idx].unsqueeze(0),
            self._info[idx]
        )
