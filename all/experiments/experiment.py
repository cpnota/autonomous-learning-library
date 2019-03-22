import os
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
from tensorboardX import SummaryWriter
from all.environments import GymEnvironment

class Experiment:
    def __init__(self, env, frames=None, episodes=None, trials=1):
        if frames is None:
            frames = np.inf
        if episodes is None:
            episodes = np.inf
        if isinstance(env, str):
            self.env = GymEnvironment(env)
        else:
            self.env = env
        self._trials = trials
        self._max_frames = frames
        self._max_episodes = episodes
        self._agent = None
        self._episode = None
        self._trial = None
        self._frames = None
        self._writer = None

    def run(
            self,
            make_agent,
            label=None,
            render=False,
            console=True,
    ):
        if label is None:
            label = make_agent.__name__
        for trial in range(self._trials):
            self._trial = trial
            self._init_trial(make_agent, label)
            while (self._episode < self._max_episodes and self._frames < self._max_frames):
                self._run_episode(render, console)

    def _init_trial(self, make_agent, label):
        self._frames = 0
        self._episode = 0
        self._writer = self._make_writer(label)
        self._agent = make_agent(self.env)

    def _run_episode(self, render, console):
        agent = self._agent
        env = self.env

        start = timer()

        # initial state
        env.reset()
        if render:
            env.render()
        env.step(agent.initial(env.state))
        returns = env.reward
        frames = 1

        # rest of episode
        while not env.should_reset:
            if render:
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
        if console:
            print("trial: %i/%i, episode: %i, frames: %i, fps: %d, returns: %d" %
                  (self._trial + 1, self._trials, self._episode, self._frames, fps, returns))

        # update state
        self._episode += 1
        self._frames += frames

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
