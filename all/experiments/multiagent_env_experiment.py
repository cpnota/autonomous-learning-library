from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment

class MultiagentEnvExperiment():
    '''
    An Experiment object for training and testing Multiagents.
    
    Args:
        preset (all.presets.Preset): A Multiagent preset.
        env (all.environments.MultiagentEnvironment): A multiagent environment.
        log_dir (str, optional): The directory in which to save the logs and model.
        name (str, optional): The name of the experiment.
        quiet (bool, optional): Whether or not to print training information.
        render (bool, optional): Whether or not to render during training.
        save_freq (int, optional): How often to save the model to disk.
        train_steps (int, optional): The number of steps for which to train.
        write_loss (bool, optional): Whether or not to log advanced loss information.
    '''
    def __init__(
            self,
            preset,
            env,
            logdir='runs',
            name='multi',
            quiet=False,
            render=False,
            save_freq=100,
            train_steps=float('inf'),
            write_loss=True,
    ):
        self._agent = self._preset.agent(writer=self._writer, train_steps=train_steps)
        self._env = env
        self._episode = 0
        self._frame = 0
        self._logdir = logdir
        self._preset = preset
        self._render = render
        self._save_freq = 100
        self._writer = ExperimentWriter(self, name, env.name, loss=write_loss, logdir=logdir)

        if render:
            self._env.render()

    '''
    Train the Multiagent for a certain number of frames or episodes.
    If both frames and episodes are specified, then the training loop will exit
    when either condition is satisfied.

    Args:
        frames (int): The maximum number of training frames.
        episodes (bool): The maximum number of training episodes.

    Returns:
        MultiagentEnvExperiment: The experiment object.
    '''
    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()
        return self

    '''
    Test the agent in eval mode for a certain number of episodes.

    Args:
        episodes (int): The number of test epsiodes.

    Returns:
        list(float): A list of all returns received during testing.
    '''
    def test(self, episodes=100):
        pass

    '''int: The number of completed training frames'''
    @property
    def frame(self):
        return self._frame

    '''int: The number of completed training episodes'''
    @property
    def episode(self):
        return self._episode

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

        # finalize the episode
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
