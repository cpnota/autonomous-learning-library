import argparse
from all.bodies import DeepmindAtariBody
from all.environments import AtariEnvironment
from all.experiments import GreedyAgent, watch


def main():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong)")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=60,
        help="Playback speed",
    )
    args = parser.parse_args()
    env = AtariEnvironment(args.env, device=args.device)
    agent = DeepmindAtariBody(GreedyAgent.load(args.dir, env))
    watch(agent, env, fps=args.fps)

if __name__ == "__main__":
    main()
