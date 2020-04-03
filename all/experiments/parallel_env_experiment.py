
import torch
import numpy as np
from all.environments import State
from .writer import ExperimentWriter
from .experiment import Experiment

class ParallelEnvExperiment(Experiment):
    def __init__(
            self,
            agent,
            env,
            render=False,
            quiet=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(agent[0].__name__, env.name, write_loss), quiet)
        make_agent, n_envs = agent
        self._env = env.duplicate(n_envs)
        self._agent = make_agent(self._env, self._writer)
        self._n_envs = n_envs
        self._render = render
        self._frame = 1
        self._episode = 1

    def train(self, frames=np.inf, episodes=np.inf):
        returns = self._reset()
        while not self._done(frames, episodes):
            self._step(returns)

    def test(self, episodes=100):
        saved_returns = [] # returns for all episodes
        returns = self._reset() # returns for current episodes

        # only save  the returns from the first n episodes started
        episodes_started = self._n_envs
        save_returns = [True] * self._n_envs

        while len(saved_returns) < episodes:
            states = State.from_list([env.state for env in self._env])
            rewards = torch.tensor(
                [env.reward for env in self._env],
                dtype=torch.float,
                device=self._env[0].device
            )
            actions = self._agent.act(states, rewards)

            for i, env in enumerate(self._env):
                if env.done:
                    returns[i] += env.reward
                    if save_returns[i]:
                        saved_returns.append(returns[i].item())
                        self._log_test_episode(len(saved_returns), returns[i].item())
                    env.reset()
                    returns[i] = 0
                    if episodes_started > episodes:
                        save_returns[i] = False
                    episodes_started += 1
                else:
                    action = actions[i]
                    if action is not None:
                        returns[i] += env.reward
                        env.step(action)
                self._frame += 1

        self._writer.add_summary('test/returns', np.mean(saved_returns), np.std(saved_returns))

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _reset(self):
        for env in self._env:
            env.reset()
        rewards = torch.zeros(
            (self._n_envs),
            dtype=torch.float,
            device=self._env[0].device
        )
        return rewards

    def _step(self, returns):
        states = State.from_list([env.state for env in self._env])
        rewards = torch.tensor(
            [env.reward for env in self._env],
            dtype=torch.float,
            device=self._env[0].device
        )
        actions = self._agent.act(states, rewards)

        for i, env in enumerate(self._env):
            self._step_env(i, env, actions[i], returns)

    def _step_env(self, i, env, action, returns):
        if env.done:
            returns[i] += env.reward
            self._log_training_episode(returns[i].item(), 0)
            env.reset()
            returns[i] = 0
            self._episode += 1
        else:
            if action is not None:
                returns[i] += env.reward
                env.step(action)
                self._frame += 1

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss)
