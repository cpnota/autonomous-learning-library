class State:
    def __init__(self, raw, done, info=None):
        self._raw = raw
        self._done = done
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
