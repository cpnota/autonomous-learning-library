# pylint: disable=unused-import
import argparse
from all.environments import GymEnvironment, PybulletEnvironment
from all.experiments import run_experiment
from all.presets import continuous


# see also: PybulletEnvironment.short_names
ENVS = {
    "mountaincar": "MountainCarContinuous-v0",
    "lander": "LunarLanderContinuous-v2",
}


def main():
    parser = argparse.ArgumentParser(description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (e.g. 'lander', 'cheetah')")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. ddpg). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=2e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment."
    )
    parser.add_argument(
        "--logdir", default='runs', help="The base logging directory."
    )
    parser.add_argument("--writer", default='tensorboard', help="The backend used for tracking experiment metrics.")
    parser.add_argument(
        '--hyperparameters',
        default=[],
        nargs='*',
        help="Custom hyperparameters, in the format hyperparameter1=value1 hyperparameter2=value2 etc."
    )
    args = parser.parse_args()

    if args.env in ENVS:
        env = GymEnvironment(args.env, device=args.device)
    elif 'BulletEnv' in args.env or args.env in PybulletEnvironment.short_names:
        env = PybulletEnvironment(args.env, device=args.device)
    else:
        env = GymEnvironment(args.env, device=args.device)

    agent_name = args.agent
    agent = getattr(continuous, agent_name)
    agent = agent.device(args.device)

    # parse hyperparameters
    hyperparameters = {}
    for hp in args.hyperparameters:
        key, value = hp.split('=')
        hyperparameters[key] = type(agent.default_hyperparameters[key])(value)
    agent = agent.hyperparameters(**hyperparameters)

    run_experiment(
        agent,
        env,
        frames=args.frames,
        render=args.render,
        logdir=args.logdir,
        writer=args.writer,
    )


if __name__ == "__main__":
    main()
