import argparse
from all.environments import AtariEnvironment
from all.experiments import run_experiment
from all.presets import atari

def main():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
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
        "--logdir", default='runs', help="The base logging directory."
    )
    args = parser.parse_args()

    env = AtariEnvironment(args.env, device=args.device)
    agent_name = args.agent
    agent = getattr(atari, agent_name)

    run_experiment(
        agent(device=args.device, last_frame=args.frames),
        env,
        args.frames,
        render=args.render,
        logdir=args.logdir
    )


if __name__ == "__main__":
    main()
