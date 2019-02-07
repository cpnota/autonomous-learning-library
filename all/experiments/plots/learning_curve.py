import numpy as np
import matplotlib.pyplot as plt

def learning_curve(results, filename=None):
    env = results["env"]
    data = results["data"]

    plt.cla()
    plt.title(env)
    for agent_name, result in data.items():
        _, episodes = result.shape
        x = np.arange(1, episodes + 1)
        y = np.mean(result, axis=0)
        if (episodes >= 1000):
            x = np.mean(x.reshape(-1, 100), axis=1)
            y = np.mean(y.reshape(-1, 100), axis=1)
        plt.plot(x, y, label=agent_name)
        plt.xlabel("episode")
        plt.ylabel("returns")
        plt.legend(loc='upper left')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
