from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment

class MultiagentEnvExperiment():
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''
    def __init__(
            self,
            preset,
            env,
            name='multi',
            train_steps=float('inf'),
            render=False,
            quiet=False,
            write_loss=True,
            save_freq=100,
    ):
        self._writer = ExperimentWriter(self, name, env.name, loss=write_loss)
        self._preset = preset
        self._agent = self._preset.agent(writer=self._writer, train_steps=train_steps)
        self._env = env
        self._render = render
        self._frame = 0
        self._episode = 0
        self._save_freq = 100

        if render:
            self._env.render()

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
        pass
        # returns = []
        # for episode in range(episodes):
        #     episode_return = self._run_test_episode()
        #     returns.append(episode_return)
        #     self._log_test_episode(episode, episode_return)
        # self._log_test(returns)
        # return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        self._env.reset()
        returns = {agent : 0 for agent in self._env.agents}

        for agent in self._env.agent_iter():
            if self._render:
                self._env.render()
            state = self._env.last()
            returns[agent] += state.reward
            action = self._agent.act(state)
            if state.done:
                self._env.step(None)
            else:
                self._env.step(action)
            self._frame += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        self._log_training_episode(returns, fps)
        self._save_model()
        self._episode += 1

    def _log_training_episode(self, returns, fps):
        print('returns: {}'.format(returns))
        print('fps: {}'.format(fps))
        for agent in self._env.agents:
            self._writer.add_evaluation('{}/returns/frame'.format(agent), returns[agent], step="frame")

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

    def _save_model(self):
        if self._episode % self._save_freq == 0:
            self._preset.save('{}/preset.pt'.format(self._writer.log_dir))

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss)
