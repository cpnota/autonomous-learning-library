import gym
from all.environments import Environment

class GymWrapper(Environment):
    def __init__(self, env):
        if (isinstance(env, str)):
            self._env = gym.make(env)
        else:
            self._env = env

        self._state = None
        self._action = None
        self._reward = None
        self._done = None
        self._info = None

    def reset(self):
        self._state = self._env.reset()
        self._done = False
        self._reward = 0
        return self._state

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        self._state = state
        self._action = action
        self._reward = reward
        self._done = done
        self._info = info
        return state, reward, done, info

    def close(self):
        return self._env.close()

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    @property
    def info(self):
        return self._info

    @property
    def env(self):
        return self._env
