import argparse
import roboschool
from all.environments import GymEnvironment
from all.experiments import Experiment
from all.presets import continuous

# some example envs
# can also enter ID directly
envs = {
    'walker': 'BipedalWalker-v2',
    'mountaincar': 'MountainCarContinuous-v0',
    'lander': 'LunarLanderContinuous-v2',
    'hopper': 'RoboschoolHopper-v1',
    'cheetah': 'RoboschoolHalfCheetah-v1'
}

def run_atari():
    parser = argparse.ArgumentParser(
        description='Run a continuous actions benchmark.')
    parser.add_argument('env', help='Name of the env (see envs)')
    parser.add_argument(
        'agent', help="Name of the agent (e.g. actor_critic). See presets for available agents.")
    parser.add_argument('--frames', type=int, default=2e6,
                        help='The number of training frames')
    parser.add_argument(
        '--device', default='cuda',
        help='The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)'
    )
    parser.add_argument(
        '--render', default=False,
        help='Whether to render the environment.'
    )
    args = parser.parse_args()

    if args.env in envs:
        env_id = envs[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id, device=args.device)
    agent_name = args.agent
    agent = getattr(continuous, agent_name)

    experiment = Experiment(
        env,
        frames=args.frames
    )
    experiment.run(agent(device=args.device), label=agent_name, render=args.render)


if __name__ == '__main__':
    run_atari()
