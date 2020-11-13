import os
import time
import torch
import gym
from all.agents import Agent


def watch(agent, env, fps=60):
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    env.render(mode="human")
    env.reset()

    while True:
        action = agent.act(env.state)
        returns += env.state.reward

        time.sleep(1 / fps)
        if env.state.done:
            print('returns:', returns)
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()


def load_and_watch(filename, env, fps=60):
    agent = torch.load(filename).test_agent()
    watch(agent, env, fps=fps)
