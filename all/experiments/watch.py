import os
import time
import torch
import gymnasium
from all.agents import Agent


def watch(agent, env, fps=60):
    action = None
    returns = 0
    env.reset()

    while True:
        env.render()
        action = agent.act(env.state)
        env.step(action)
        returns += env.state.reward

        if env.state.done:
            print('returns:', returns)
            env.reset()
            returns = 0

        time.sleep(1 / fps)


def load_and_watch(filename, env, fps=60):
    agent = torch.load(filename).test_agent()
    watch(agent, env, fps=fps)
