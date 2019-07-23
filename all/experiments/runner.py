from abc import ABC, abstractmethod
from timeit import default_timer as timer
import numpy as np
import torch
from all.environments import State
from .writer import ExperimentWriter

class EnvRunner(ABC):
    def __init__(
            self,
            agent,
            env,
            agent_name=None,
            env_name=None,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        self._env = env
        self._agent_name = agent_name or agent.__name__
        self._env_name = env_name or self._env.name
        self._max_frames = frames
        self._max_episodes = episodes
        self._render = render
        self._quiet = quiet
        self._writer = self._make_writer(write_loss)
        self._agent = agent(env, self._writer)
        self.run()

    @abstractmethod
    def run(self):
        pass

    def _done(self):
        return (
            self._writer.frames > self._max_frames or 
            self._writer.episodes > self._max_episodes
        )

    def _log(self, returns, fps):
        if not self._quiet:
            print("episode: %i, frames: %i, fps: %d, returns: %d" %
                  (self._writer.episodes, self._writer.frames, fps, returns))
        self._writer.add_evaluation('returns-by-episode', returns, step="episode")
        self._writer.add_evaluation('returns-by-frame', returns, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")

    def _make_writer(self, write_loss):
        return ExperimentWriter(self._agent_name, self._env_name, loss=write_loss)

class SingleEnvRunner(EnvRunner):
    def run(self):
        while not self._done():
            self._run_episode()

    def _run_episode(self):
        start_time = timer()
        start_frames = self._writer.frames
        returns = self._run_until_terminal_state()
        end_time = timer()
        fps = (self._writer.frames - start_frames) / (end_time - start_time)
        self._log(returns, fps)
        self._writer.episodes += 1

    def _run_until_terminal_state(self):
        agent = self._agent
        env = self._env

        env.reset()
        returns = 0
        action = agent.act(env.state, env.reward)

        while not env.done:
            self._writer.frames += 1
            if self._render:
                env.render()
            env.step(action)
            returns += env.reward
            action = agent.act(env.state, env.reward)

        return returns

class ParallelEnvRunner(EnvRunner):
    def __init__(self, agent, env, **kwargs):
        make_agent, n_envs = agent
        envs = env.duplicate(n_envs)
        self._n_envs = n_envs
        self._returns = None
        self._start_time = None
        super().__init__(make_agent, envs, env_name=env.name, **kwargs)

    def run(self):
        self._reset()
        while not self._done():
            self._step()

    def _reset(self):
        for env in self._env:
            env.reset()
        self._returns = torch.zeros(
            (self._n_envs),
            dtype=torch.float,
            device=self._env[0].device
        )
        self._start_time = timer()

    def _step(self):
        states = State.from_list([env.state for env in self._env])
        rewards = torch.tensor(
            [env.reward for env in self._env],
            dtype=torch.float,
            device=self._env[0].device
        )
        actions = self._agent.act(states, rewards)

        for i, env in enumerate(self._env):
            self._step_env(i, env, actions[i])

    def _step_env(self, i, env, action):
        if env.done:
            self._returns[i] += env.reward
            end_time = timer()
            fps = self._writer.frames / (end_time - self._start_time)
            self._log(self._returns[i], fps)
            env.reset()
            self._returns[i] = 0
            self._writer.episodes += 1
        else:
            if action is not None:
                self._returns[i] += env.reward
                env.step(action)
                self._writer.frames += 1
