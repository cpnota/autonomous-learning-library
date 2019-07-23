import argparse
from all.environments import GymEnvironment
from all.experiments import Experiment
from all.presets import classic_control


def run_classic():
    parser = argparse.ArgumentParser(description="Run a classic control benchmark.")
    parser.add_argument("env", help="Name of the env (e.g. CartPole-v1)")
    parser.add_argument(
        "agent",
        help="Name of the agent (e.g. sarsa). See presets for available agents.",
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="The number of training frames"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()

    env = GymEnvironment(args.env, device=args.device)
    agent_name = args.agent
    agent = getattr(classic_control, agent_name)

    Experiment(agent(device=args.device), env, episodes=args.episodes)


if __name__ == "__main__":
    run_classic()
