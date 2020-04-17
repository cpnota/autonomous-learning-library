# pylint: disable=unused-import
import argparse
import pybullet
import pybullet_envs
from all.bodies import TimeFeature
from all.environments import GymEnvironment
from all.experiments import GreedyAgent, watch
from continuous import ENVS


def main():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
    parser.add_argument("env", help="ID of the Environment")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
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
    agent = TimeFeature(GreedyAgent.load(args.dir, env))
    watch(agent, env, fps=args.fps)

if __name__ == "__main__":
    main()
