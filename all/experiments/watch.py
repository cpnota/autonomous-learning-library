import sys
import time

import torch


def watch(agent, env, fps=60, n_episodes=sys.maxsize):
    action = None
    returns = 0
    env.reset()

    for _ in range(n_episodes):
        env.render()
        action = agent.act(env.state)
        env.step(action)
        returns += env.state.reward

        if env.state.done:
            print("returns:", returns)
            env.reset()
            returns = 0

        time.sleep(1 / fps)


def load_and_watch(filename, env, fps=60, n_episodes=sys.maxsize):
    agent = torch.load(filename).test_agent()
    watch(agent, env, fps=fps, n_episodes=n_episodes)
