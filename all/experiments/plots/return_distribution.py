import numpy as np
import matplotlib.pyplot as plt

def return_distribution(results, filename=None):
    env = results["env"]
    data = results["data"]

    plt.cla()
    plt.title(env)
    for agent_name, result in data.items():
        trials, _ = result.shape
        x = np.arange(trials) * 100 / trials
        y = np.sort(np.mean(result, axis=1))
        plt.plot(x, y, label=agent_name)
        plt.xlabel("percentile")
        plt.ylabel("average returns")
        plt.legend(loc='upper left')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
