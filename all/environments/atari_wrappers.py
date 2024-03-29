"""
A subset of Atari wrappers modified from:
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
Other behaviors were implemented as Bodies.
"""

import os

import numpy as np

os.environ.setdefault("PATH", "")

import cv2
import gymnasium

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gymnasium.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gymnasium.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gymnasium.Wrapper):
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.

        Important: This was modified to also fire on lives lost.
        """
        gymnasium.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, info = self.fire()
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.lost_life():
            obs, info = self.fire()
        self.lives = self.env.unwrapped.ale.lives()
        return obs, reward, terminated, truncated, info

    def fire(self):
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info

    def lost_life(self):
        lives = self.env.unwrapped.ale.lives()
        return lives < self.lives and lives > 0


class MaxAndSkipEnv(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gymnasium.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gymnasium.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return np.moveaxis(obs, -1, 0)


class LifeLostEnv(gymnasium.Wrapper):
    def __init__(self, env):
        """
        Modified wrapper to add a "life_lost" key to info.
        This allows the agent Body to make the episode as done
        if it desires.
        """
        gymnasium.Wrapper.__init__(self, env)
        self.lives = 0

    def reset(self):
        obs, _ = self.env.reset()
        self.lives = 0
        return obs, {"life_lost": False}

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        life_lost = lives < self.lives and lives > 0
        self.lives = lives
        info = {"life_lost": life_lost}
        return obs, reward, terminated, truncated, info
