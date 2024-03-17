import argparse

from all.environments import MujocoEnvironment
from all.experiments import load_and_watch


def main():
    parser = argparse.ArgumentParser(description="Watch a mujoco agent.")
    parser.add_argument("env", help="ID of the Environment")
    parser.add_argument("filename", help="File where the model was saved.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=120,
        help="Playback speed",
    )
    args = parser.parse_args()
    env = MujocoEnvironment(args.env, device=args.device, render_mode="human")
    load_and_watch(args.filename, env, fps=args.fps)


if __name__ == "__main__":
    main()
