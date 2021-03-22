import os
import numpy as np
import matplotlib.pyplot as plt


def plot_returns_100(runs_dir, timesteps=-1):
    data = load_returns_100_data(runs_dir)
    lines = {}
    fig, axes = plt.subplots(1, len(data))
    if len(data) == 1:
        axes = [axes]
    for i, env in enumerate(sorted(data.keys())):
        ax = axes[i]
        subplot_returns_100(ax, env, data[env], lines, timesteps=timesteps)
    fig.legend(list(lines.values()), list(lines.keys()), loc="center right")
    plt.show()


def load_returns_100_data(runs_dir):
    data = {}

    def add_data(agent, env, file):
        if env not in data:
            data[env] = {}
        data[env][agent] = np.genfromtxt(file, delimiter=",").reshape((-1, 3))

    for agent_dir in os.listdir(runs_dir):
        agent = agent_dir.split("_")[0]
        agent_path = os.path.join(runs_dir, agent_dir)
        if os.path.isdir(agent_path):
            for env in os.listdir(agent_path):
                env_path = os.path.join(agent_path, env)
                if os.path.isdir(env_path):
                    returns100path = os.path.join(env_path, "returns100.csv")
                    if os.path.exists(returns100path):
                        add_data(agent, env, returns100path)

    return data


def subplot_returns_100(ax, env, data, lines, timesteps=-1):
    for agent in data:
        agent_data = data[agent]
        x = agent_data[:, 0]
        mean = agent_data[:, 1]
        std = agent_data[:, 2]

        if timesteps > 0:
            x[-1] = timesteps

        if agent in lines:
            ax.plot(x, mean, label=agent, color=lines[agent].get_color())
        else:
            line, = ax.plot(x, mean, label=agent)
            lines[agent] = line
        ax.fill_between(
            x, mean + std, mean - std, alpha=0.2, color=lines[agent].get_color()
        )
        ax.set_title(env)
        ax.set_xlabel("timesteps")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 5))
