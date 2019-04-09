import os
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
from tensorboardX import SummaryWriter
from all.environments import GymEnvironment

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
        self._init_trial(label, render, console)
        if isinstance(make_agent, tuple):
            make, n_envs = make_agent
            self._run_multi(make, n_envs)
        else:
            self._run_single(make_agent)

    def _init_trial(self, label, render, console):
        if label is None:
            label = 'agent'
        self._frames = 0
        self._episode = 0
        self._render = render
        self._console = console
        self._writer = self._make_writer(label)

    def _run_multi(self, make_agent, n_envs):
        raise Exception('Not implemented.')

    def _run_single(self, make_agent):
        self._agent = make_agent(self.env)
        while not self._done():
            self._run_episode()

    def _run_episode(self):
        env = self.env
        agent = self._agent

        start = timer()

        # initial state
        env.reset()
        if self._render:
            env.render()
        env.step(agent.initial(env.state))
        returns = env.reward
        frames = 1

        # rest of episode
        while not env.done:
            if self._render:
                env.render()
            env.step(agent.act(env.state, env.reward))
            returns += env.reward
            frames += 1

        # terminal state
        agent.terminal(env.reward)

        # log info
        end = timer()
        fps = frames / (end - start)
        self._log(returns, fps)
        if self._console:
            print("episode: %i, frames: %i, fps: %d, returns: %d" %
                  (self._episode, self._frames, fps, returns))

        # update state
        self._episode += 1
        self._frames += frames

    def _done(self):
        return self._frames > self._max_frames or self._episode > self._max_episodes

    def _log(self, returns, fps):
        self._writer.add_scalar(
            self.env.name + '/returns/eps', returns, self._episode)
        self._writer.add_scalar(
            self.env.name + '/returns/frames', returns, self._frames)
        self._writer.add_scalar(self.env.name + '/fps', fps, self._frames)

    def _make_writer(self, label):
        current_time = str(datetime.now())
        log_dir = os.path.join(
            'runs', label + " " + current_time
        )
        return SummaryWriter(log_dir=log_dir)
