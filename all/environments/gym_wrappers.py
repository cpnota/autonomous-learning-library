import gymnasium


class NoInfoWrapper(gymnasium.Wrapper):
    """
    Wrapper to suppress info and simply return a dict.
    This prevents State.from_gym() from create keys.
    """

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        *obs, info = self.env.step(action)
        return *obs, {}
