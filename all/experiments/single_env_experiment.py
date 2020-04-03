from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment

class SingleEnvExperiment(Experiment):
    def __init__(
            self,
            agent,
            env,
            render=False,
            quiet=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(agent.__name__, env.name, write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, episodes=100):
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode()
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._writer.add_summary('test/returns', np.mean(returns), np.std(returns))

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # run the episode
        action, returns = self._reset()
        while not self._env.done:
            self._frame += 1
            action, returns = self._step(action, returns)

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

    def _run_test_episode(self):
        action, returns = self._reset()
        while not self._env.done:
            action, returns = self._step(action, returns)
        return returns

    def _reset(self):
        self._env.reset()
        return self._agent.act(self._env.state, self._env.reward), 0

    def _step(self, action, returns):
        if self._render:
            self._env.render()
        self._env.step(action)
        action = self._agent.act(self._env.state, self._env.reward)
        return action, returns + self._env.reward

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss)
