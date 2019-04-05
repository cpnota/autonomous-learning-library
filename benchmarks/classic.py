import argparse
from all.environments import GymEnvironment
from all.presets import classic_control
from runner import run

def run_atari():
    parser = argparse.ArgumentParser(description='Run a classic control benchmark.')
    parser.add_argument('env', help='Name of the env (e.g. CartPole-v1)')
    parser.add_argument('agent', help="Name of the agent (e.g. sarsa). See presets for available agents.")
    # parser.add_argument('device', default='cpu', help='The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)')
    parser.add_argument('--episodes', type=int, default=1000, help='The number of training frames')
    args = parser.parse_args()

    env = GymEnvironment(args.env)
    agent_name = args.agent
    agent = getattr(classic_control, agent_name)
    episodes = args.episodes

    run(agent_name, agent(), env, episodes=episodes)

if __name__ == '__main__':
    run_atari()
