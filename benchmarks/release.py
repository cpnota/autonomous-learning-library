import argparse
from all.environments import AtariEnvironment, GymEnvironment
from all.presets import atari, classic_control
from runner import run


def run_release():
    parser = argparse.ArgumentParser(
        description='Run a release benchmark of all the included algorithms.')
    parser.add_argument('device', default='cpu',
                        help='The name of the device to run the agents on (e.g. cpu, cuda, cuda:0)')
    args = parser.parse_args()

    device = args.device
    print("Running release benchmarks on device: " + device)

    for agent_name in classic_control.__all__:
        env = GymEnvironment('CartPole-v1')
        agent = getattr(classic_control, agent_name)
        run(agent_name, agent(), env, episodes=1000)

    for agent_name in atari.__all__:
        env = AtariEnvironment('Pong', device=device)
        agent = getattr(atari, agent_name)
        run(agent_name, agent(device=device), env, frames=10e6)


if __name__ == '__main__':
    run_release()
