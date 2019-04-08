import os
import torch
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
from tensorboardX import SummaryWriter
from all.presets.classic_control import a2c
from all.environments import GymEnvironment

def run():
    n = 8
    envs = []
    for i in range(n):
        envs.append(GymEnvironment('CartPole-v0'))
    agent = a2c()(envs[0])
    frames = 0
    [env.reset() for env in envs]
    returns = torch.zeros((n)).float()

    while(frames < 50000):
        states = [env.state for env in envs]
        rewards = torch.tensor([env.reward for env in envs]).float()
        returns += rewards
        actions = agent.act(states, rewards)
        for i in range(len(envs)):
            if envs[i].done:
                print('returns', returns[i].item())
                returns[i] = 0
                envs[i].reset()
            else:
                envs[i].step(actions[i])
        frames += n

if __name__ == '__main__':
    run()
