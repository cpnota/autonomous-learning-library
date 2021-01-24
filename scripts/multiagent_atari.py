import argparse
from all.environments import MultiagentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import multiagent_atari


def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=40e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", type=bool, default=False, help="Render the environment."
    )
    parser.add_argument(
        "--writer", default='tensorboard', help="The backend used for tracking experiment metrics."
    )
    args = parser.parse_args()

    env = MultiagentAtariEnv(args.env, device=args.device)
    agent_name = args.agent
    agent = getattr(multiagent_atari, agent_name)
    experiment = MultiagentEnvExperiment(agent(device=args.device), env, write_loss=False, writer=args.writer)
    experiment.train(frames=args.frames)


if __name__ == "__main__":
    main()
