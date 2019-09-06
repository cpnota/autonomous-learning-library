import argparse
from all.environments import AtariEnvironment
from all.experiments import Experiment
from all.presets import atari

def run_atari():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong)")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--frames", type=int, default=200e6, help="The number of training frames"
    )
    args = parser.parse_args()

    env = AtariEnvironment(args.env, device=args.device)
    agent_name = args.agent
    agent = getattr(atari, agent_name)

    Experiment(agent(device=args.device), env, frames=args.frames)


if __name__ == "__main__":
    run_atari()
