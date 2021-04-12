# pylint: disable=unused-import
import argparse
from all.bodies import TimeFeature
from all.environments import GymEnvironment, PybulletEnvironment
from all.experiments import load_and_watch
from .continuous import ENVS


def main():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
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

    if args.env in ENVS:
        env = GymEnvironment(args.env, device=args.device)
    elif 'BulletEnv' in args.env or args.env in PybulletEnvironment.short_names:
        env = PybulletEnvironment(args.env, device=args.device)
    else:
        env = GymEnvironment(args.env, device=args.device)

    load_and_watch(args.filename, env, fps=args.fps)


if __name__ == "__main__":
    main()
