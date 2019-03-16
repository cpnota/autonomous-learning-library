import json
from timeit import default_timer as timer
import numpy as np
from tensorboardX import SummaryWriter
from all.environments import GymEnvironment

class Experiment:
    def __init__(self, env, episodes=200, trials=100):
        if isinstance(env, str):
            self.env = GymEnvironment(env)
            self.env_name = env
        else:
            self.env = env
            self.env_name = env.__class__.__name__
        self.episodes = episodes
        self.trials = trials
        self.data = {}
        self._writer = None

    def run(
            self,
            make_agent,
            agent_name=None,
            print_every=np.inf,
            render=False
    ):
        self._writer = self._make_writer()
        agent_name = make_agent.__name__ if agent_name is None else agent_name
        self.data[agent_name] = np.zeros((0, self.episodes))
        frames = 0

        for trial in range(self.trials):
            agent = make_agent(self.env)
            self.data[agent_name] = np.vstack((self.data[agent_name], np.zeros(self.episodes)))
            for episode in range(self.episodes):
                returns, _frames = run_episode(agent, self.env, render)
                frames += _frames
                self.data[agent_name][trial][episode] = returns
                self._log(trial, episode, returns, frames, print_every)

        return self.data[agent_name]

    def _log(self, trial, episode, returns, frames, print_every):
        episode_number = trial * self.episodes + episode + 1
        self._writer.add_scalar('returns', returns, episode)
        if episode_number % print_every == 0:
            print("trial: %i/%i, episode: %i/%i, frames: %i, returns: %d" %
                  (trial + 1, self.trials, episode + 1, self.episodes, frames, returns))

    def _make_writer(self):
        return SummaryWriter()


def run_episode(agent, env, render=False):
    start = timer()
    env.reset()
    if render:
        env.render()
    env.step(agent.initial(env.state))
    returns = env.reward
    frames = 1
    while not env.should_reset:
        if render:
            env.render()
        env.step(agent.act(env.state, env.reward))
        returns += env.reward
        frames += 1
    agent.terminal(env.reward)
    end = timer()
    print('episode fps:', frames / (end - start))
    return returns, frames
