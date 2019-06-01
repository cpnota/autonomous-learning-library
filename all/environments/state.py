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
    def info(self):
        return self._info
