# pylint: disable=unused-import
import argparse
import pybullet
import pybullet_envs
from all.bodies import TimeFeature
from all.environments import GymEnvironment
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
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id, device=args.device)
    load_and_watch(args.filename, env, fps=args.fps)


if __name__ == "__main__":
    main()
