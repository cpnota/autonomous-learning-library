from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment

class SingleEnvExperiment(Experiment):
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''
    def __init__(
            self,
            agent,
            env,
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(logdir, agent.__name__, env.name, write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, episodes=100):
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode()
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._log_test(returns)
        return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        action = self._agent.act(state)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = self._agent.act(state)
            returns += state.reward
            self._frame += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

    def _run_test_episode(self):
        # initialize the episode
        state = self._env.reset()
        action = self._agent.eval(state)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = self._agent.eval(state)
            returns += state.reward

        return returns

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
