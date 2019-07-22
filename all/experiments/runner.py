from abc import ABC, abstractmethod
from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter

class EnvRunner(ABC):
    def __init__(
            self,
            agent,
            env,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        self._label = agent.__name__
        self._env = env
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
        return ExperimentWriter(self._label, self._env.name, loss=write_loss)

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
    def run(self):
        pass
