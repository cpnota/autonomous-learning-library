import numpy as np
import matplotlib.pyplot as plt
from all.environments import GymWrapper


class LearningCurve:
    results = None

    def run(
            self,
            make_agent,
            env,
            episodes=200,
            trials=100
    ):
        agent = None
        results = np.zeros((trials, episodes))

        if isinstance(env, str):
            env = GymWrapper(env)

        print('Running learning curve experiment...')
        for trial in range(trials):
            agent = make_agent(env)
            for episode in range(episodes):
                returns = run_episode(agent, env)
                results[trial][episode] = returns
        print('Learning curve experiment finished!')

        self.results = results
        return results

    def plot(self, results=None):
        if results is None:
            results = self.results
        (trials, episodes) = results.shape
        x = np.arange(1, episodes + 1)
        y = np.mean(results, axis=0)
        stderr = np.std(results, axis=0) / np.sqrt(trials)
        plt.errorbar(x, y, stderr)
        plt.xlabel("episode")
        plt.ylabel("returns")
        plt.show()


def run_episode(agent, env):
    env.reset()
    agent.new_episode(env)
    returns = 0
    while not env.done:
        agent.act()
        returns += env.reward
    return returns
