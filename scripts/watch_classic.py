import argparse
from all.environments import GymEnvironment
from all.experiments import load_and_watch

def main():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the environment (e.g. RoboschoolHalfCheetah-v1")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    env = GymEnvironment(args.env, device=args.device)
    load_and_watch(args.dir, env)

if __name__ == "__main__":
    main()
