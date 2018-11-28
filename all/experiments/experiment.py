import json
import numpy as np
import matplotlib.pyplot as plt
from .plots import learning_curve
from all.environments import GymWrapper


class Experiment:
    def __init__(self, env, episodes=200, trials=100):
        if isinstance(env, str):
            self.env = GymWrapper(env)
            self.env_name = env
        else:
            self.env = env
            self.env_name = env.__class__.__name__
        self.episodes = episodes
        self.trials = trials
        self.data = {}

    @property
    def results(self):
        return {
            "env": self.env_name,
            "episodes": self.episodes,
            "trials": self.trials,
            "data": self.data
        }

    def run(
            self,
            make_agent,
            agent_name=None,
            print_every=np.inf,
            plot_every=np.inf,
            plot=learning_curve
    ):
        agent_name = make_agent.__name__ if agent_name is None else agent_name
        self.data[agent_name] = np.zeros((self.trials, self.episodes))

        print('Running: %s on %s...' % (agent_name, self.env_name))
        for trial in range(self.trials):
            agent = make_agent(self.env)
            for episode in range(self.episodes):
                returns = run_episode(agent, self.env)
                self.data[agent_name][trial][episode] = returns
                self.monitor(trial, episode, returns, print_every, plot_every, plot)
        print('Finished %s on %s!' % (agent_name, self.env_name))

        return self.data[agent_name]

    def plot(self, plot=learning_curve):
        plot(self.results)

    def monitor(self, trial, episode, returns, print_every, plot_every, plot):
        episode_number = trial * self.episodes + episode + 1
        if episode_number % print_every == 0:
            print("trial: %i/%i, episode: %i/%i, returns: %d" %
                  (trial + 1, self.trials, episode + 1, self.episodes, returns))
        if episode_number % plot_every == 0:
            plt.ion()
            self.plot(plot)
            plt.pause(0.0001)
            plt.ioff()

    def save(self, filename):
        results = self.results
        results["data"] = {k:v.tolist() for (k, v) in results["data"].items()}
        with open(filename, 'w') as outfile:
            json.dump(results, outfile)

    def load(filename):
        with open(filename) as infile:
            results = json.load(infile)
        results["data"] = {k:np.array(v) for (k, v) in results["data"].items()}
        return results

def run_episode(agent, env):
    env.reset()
    agent.new_episode(env)
    returns = 0
    while not env.done:
        agent.act()
        returns += env.reward
    return returns
