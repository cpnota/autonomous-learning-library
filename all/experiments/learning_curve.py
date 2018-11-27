import json
import matplotlib.pyplot as plt
import numpy as np
from all.environments import GymWrapper


class LearningCurve:
    def __init__(self, env, episodes=200, trials=100):
        if isinstance(env, str):
            self.env = GymWrapper(env)
            self.env_name = env
        else:
            self.env = env
            self.env_name = env.__class__.__name__
        self.episodes = episodes
        self.trials = trials
        self.results = {}

    def run(
            self,
            make_agent,
            agent_name=None,
            print_every=np.inf
    ):
        agent = None
        if agent_name is None:
            agent_name = make_agent.__name__
        self.results[agent_name] = np.zeros((self.trials, self.episodes))

        print('Generating learning curve for ' + agent_name + "...")
        for trial in range(self.trials):
            agent = make_agent(self.env)
            for episode in range(self.episodes):
                returns = run_episode(agent, self.env)
                self.log(trial, episode, returns, print_every)
                self.results[agent_name][trial][episode] = returns

        return self.results[agent_name]

    def plot(self):
        print("Plotting learning curve...")
        plt.title(self.env_name)
        for agent_name, results in self.results.items():
            x = np.arange(1, self.episodes + 1)
            y = np.mean(results, axis=0)
            plt.plot(x, y, label=agent_name)
            plt.xlabel("episode")
            plt.ylabel("returns")
            plt.legend(loc='upper left')
        plt.show()

    def save(self, filename):
        data = {
            "env": self.env_name,
            "episodes": self.episodes,
            "trials": self.trials,
            "results":  {k:v.tolist() for (k, v) in self.results.items()}
        }
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load(self, filename):
        with open(filename) as infile:
            data = json.load(infile)

        self.env_name = data["env"]
        self.episodes = data["episodes"]
        self.trials = data["trials"]
        self.results = {k:np.array(v) for (k, v) in data["results"].items()}

    def log(self, trial, episode, returns, print_every):
        episode_number = trial * self.episodes + episode + 1
        if episode_number % print_every == 0:
            print("trial: %i/%i, episode: %i/%i, returns: %d" %
                  (trial + 1, self.trials, episode + 1, self.episodes, returns))

def run_episode(agent, env):
    env.reset()
    agent.new_episode(env)
    returns = 0
    while not env.done:
        agent.act()
        returns += env.reward
    return returns
