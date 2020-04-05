import gym
import numpy as np
import torch
from .abstract import Environment
from .state import State
gym.logger.set_level(40)

class GymEnvironment(Environment):
    def __init__(self, env, device=torch.device('cpu')):
        self._name = env
        if isinstance(env, str):
            env = gym.make(env)
        self._env = env
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

        # lazy init for slurm
        self._init = False
        self._done_mask = None
        self._not_done_mask = None

    @property
    def name(self):
        return self._name

    def reset(self):
        self._lazy_init()
        state = self._env.reset()
        self._state = self._make_state(state, 0)
        self._reward = 0
        self._done = False
        return self._state

    def step(self, action):
        state, reward, done, info = self._env.step(self._convert(action))
        self._state = self._make_state(state, done, info)
        self._action = action
        self._reward = reward
        self._done = done
        return self._state, self._reward

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self, n):
        return [GymEnvironment(self._name, device=self.device) for _ in range(n)]

    @property
    def state_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

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
        return self._state.info

    @property
    def env(self):
        return self._env

    @property
    def device(self):
        return self._device

    def _lazy_init(self):
        if not self._init:
            # predefining these saves performance on tensor creation
            # it actually makes a noticable difference :p
            self._done_mask = torch.tensor(
                [0],
                dtype=torch.uint8,
                device=self._device
            )
            self._not_done_mask = torch.tensor(
                [1],
                dtype=torch.uint8,
                device=self._device
            )
            self._init = True

    def _make_state(self, raw, done, info=None):
        '''Convert numpy array into State'''
        return State(
            torch.from_numpy(
                np.array(
                    raw,
                    dtype=self.state_space.dtype
                )
            ).unsqueeze(0).to(self._device),
            self._done_mask if done else self._not_done_mask,
            [info]
        )

    def _convert(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.item()
        if isinstance(self.action_space, gym.spaces.Box):
            return action.cpu().detach().numpy().reshape(-1)
        raise TypeError("Unknown action space type")
