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
    episodes = 0
    [env.reset() for env in envs]
    returns = torch.zeros((n)).float()
    writer = make_writer('a2c')

    while(frames < 50000):
        states = [env.state for env in envs]
        rewards = torch.tensor([env.reward for env in envs]).float()
        actions = agent.act(states, rewards)
        for i in range(len(envs)):
            if envs[i].done:
                returns[i] += rewards[i]
                print('frames:', frames, 'returns:', returns[i].item())
                envs[i].reset()
                writer.add_scalar(envs[i].name + '/returns/eps', returns[i], episodes)
                writer.add_scalar(envs[i].name + '/returns/frames', returns[i], frames)
                returns[i] = 0
                episodes += 1
            else:
                if actions[i] is not None:
                    returns[i] += rewards[i]
                    envs[i].step(actions[i])
                    frames += 1

def make_writer(label):
    current_time = str(datetime.now())
    log_dir = os.path.join(
        'runs', label + " " + current_time
    )
    return SummaryWriter(log_dir=log_dir)


if __name__ == '__main__':
    run()
