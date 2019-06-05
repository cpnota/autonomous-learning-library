from timeit import default_timer as timer
import numpy as np
import torch
from all.environments import GymEnvironment, State
from .writer import ExperimentWriter


class Experiment:
    def __init__(self, env, frames=None, episodes=None):
        if frames is None:
            frames = np.inf
        if episodes is None:
            episodes = np.inf
        if isinstance(env, str):
            self.env = GymEnvironment(env)
        else:
            self.env = env
        self._max_frames = frames
        self._max_episodes = episodes
        self._agent = None
        self._episode = None
        self._frames = None
        self._writer = None
        self._render = None
        self._console = None

    def run(
            self,
            make_agent,
            label=None,
            render=False,
            console=True,
    ):
        if isinstance(make_agent, tuple):
            make, n_envs = make_agent
            self._init_trial(make, label, render, console)
            self._run_multi(make, n_envs)
        else:
            self._init_trial(make_agent, label, render, console)
            self._run_single(make_agent)

    def _init_trial(self, make_agent, label, render, console):
        if label is None:
            label = make_agent.__name__
        self._frames = 0
        self._episode = 1
        self._render = render
        self._console = console
        self._writer = self._make_writer(label)

    def _run_single(self, make_agent):
        self._agent = make_agent(self.env, writer=self._writer)
        while not self._done():
            self._run_episode()

    def _run_episode(self):
        # setup
        env = self.env
        agent = self._agent
        start = timer()
        start_frames = self._frames
        returns = 0

        # run episode
        env.reset()
        while not env.done:
            if self._render:
                env.render()
            env.step(agent.act(env.state, env.reward))
            returns += env.reward
            self._frames += 1
            self._writer.frames = self._frames
        agent.act(env.state, env.reward)

        # cleanup and logging
        end = timer()
        fps = (self._frames - start_frames) / (end - start)
        self._log(returns, fps)
        self._episode += 1
        self._writer.episodes = self._episode

    def _run_multi(self, make_agent, n_envs):
        envs = self.env.duplicate(n_envs)
        agent = make_agent(envs, writer=self._writer)
        for env in envs:
            env.reset()
        returns = torch.zeros((n_envs)).float().to(self.env.device)
        start = timer()
        while not self._done():
            states = State.from_list([env.state for env in envs])
            rewards = torch.tensor([env.reward for env in envs]).float().to(self.env.device)
            actions = agent.act(states, rewards)
            for i, env in enumerate(envs):
                if env.done:
                    end = timer()
                    fps = self._frames / (end - start)
                    returns[i] += rewards[i]
                    self._log(returns[i], fps)
                    env.reset()
                    returns[i] = 0
                    self._episode += 1
                    self._writer.episodes = self._episode
                else:
                    if actions[i] is not None:
                        returns[i] += rewards[i]
                        env.step(actions[i])
                        self._frames += 1
                        self._writer.frames = self._frames

    def _done(self):
        return self._frames > self._max_frames or self._episode > self._max_episodes

    def _log(self, returns, fps):
        if self._console:
            print("episode: %i, frames: %i, fps: %d, returns: %d" %
                  (self._episode, self._frames, fps, returns))
        self._writer.add_evaluation('returns-by-episode', returns, step="episode")
        self._writer.add_evaluation('returns-by-frame', returns, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")

    def _make_writer(self, label):
        return ExperimentWriter(label, self.env.name)
