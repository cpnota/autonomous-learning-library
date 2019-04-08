import os
from datetime import datetime
from timeit import default_timer as timer
import torch
from tensorboardX import SummaryWriter
from all.presets.atari import a2c
from all.environments import AtariEnvironment

def run():
    n = 50
    envs = []
    device = torch.device('cuda')
    for i in range(n):
        envs.append(AtariEnvironment('Pong', device=device))
    agent = a2c(device=device)(envs)
    frames = 0
    episodes = 0
    for env in envs:
        env.reset()
    returns = torch.zeros((n)).float().to(device)
    writer = make_writer('a2c')
    start = timer()

    while frames < 40e6:
        states = [env.state for env in envs]
        rewards = torch.tensor([env.reward for env in envs]).float().to(device)
        actions = agent.act(states, rewards)
        for i, env in enumerate(envs):
            if env.done:
                end = timer()
                fps = frames / (end - start)
                returns[i] += rewards[i]
                print('frames:', frames, 'fps', fps, 'returns:', returns[i].item())
                env.reset()
                writer.add_scalar(env.name + '/returns/eps', returns[i], episodes)
                writer.add_scalar(env.name + '/returns/frames', returns[i], frames)
                writer.add_scalar(env.name + '/fps', fps, frames)
                returns[i] = 0
                episodes += 1
            else:
                if actions[i] is not None:
                    returns[i] += rewards[i]
                    env.step(actions[i])
                    frames += 1

def make_writer(label):
    current_time = str(datetime.now())
    log_dir = os.path.join(
        'runs', label + " " + current_time
    )
    return SummaryWriter(log_dir=log_dir)

if __name__ == '__main__':
    run()
