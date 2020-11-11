import argparse
from all.bodies import DeepmindAtariBody
from all.environments import AtariEnvironment
from all.experiments import load_and_watch


def main():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong)")
    parser.add_argument("filename", help="File where the model was saved.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=60,
        help="Playback speed",
    )
    args = parser.parse_args()
    env = AtariEnvironment(args.env, device=args.device)
    load_and_watch(args.filename, env, fps=args.fps)


if __name__ == "__main__":
    main()
