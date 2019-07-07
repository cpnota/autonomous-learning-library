import argparse
from all.environments import GymEnvironment
from all.experiments import Experiment
from all.presets import box2d

envs = {
    'walker': 'BipedalWalker-v2',
    'walker_hard': 'BipedalWalkerHardcore-v2',
    'car': 'CarRacing-v0',
    'lander': 'LunarLanderContinuous-v2',
}

def run_atari():
    parser = argparse.ArgumentParser(
        description='Run a Box2D benchmark.')
    parser.add_argument('env', help='Name of the env (walker, walker_hard, car, lander)')
    parser.add_argument(
        'agent', help="Name of the agent (e.g. actor_critic). See presets for available agents.")
    parser.add_argument('--frames', type=int, default=100000,
                        help='The number of training frames')
    parser.add_argument(
        '--device', default='cuda',
        help='The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)'
    )
    args = parser.parse_args()

    env = GymEnvironment(envs[args.env], device=args.device)
    agent_name = args.agent
    agent = getattr(box2d, agent_name)

    experiment = Experiment(
        env,
        frames=args.frames
    )
    experiment.run(agent(device=args.device), label=agent_name)


if __name__ == '__main__':
    run_atari()
