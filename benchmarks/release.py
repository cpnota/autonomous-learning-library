import argparse
from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import Experiment
from all.presets import atari, classic_control


def run_release():
    parser = argparse.ArgumentParser(
        description='Run a release benchmark of all the included algorithms.')
    parser.add_argument('--device', default='cuda',
                        help='The name of the device to run the agents on (e.g. cpu, cuda, cuda:0)')
    args = parser.parse_args()

    device = args.device
    print("Running release benchmarks on device: " + device)

    for agent_name in classic_control.__all__:
        env = GymEnvironment('CartPole-v0', device=device)
        agent = getattr(classic_control, agent_name)
        experiment = Experiment(
            env,
            episodes=1000
        )
        experiment.run(agent(device=device), label=agent_name)

    for agent_name in atari.__all__:
        env = AtariEnvironment('Pong', device=device)
        agent = getattr(atari, agent_name)
        experiment = Experiment(
            env,
            frames=20e6
        )
        experiment.run(agent(device=device), label=agent_name)


if __name__ == '__main__':
    run_release()
