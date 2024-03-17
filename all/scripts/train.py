import argparse

from all.experiments import run_experiment


def train(
    presets,
    env_constructor,
    description="Train an RL agent",
    env_help="Name of the environment (e.g., 'CartPole-v0')",
    default_frames=40e6,
):
    # parse command line args
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("env", help=env_help)
    parser.add_argument(
        "agent",
        help="Name of the agent (e.g. 'dqn'). See presets for available agents.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=default_frames,
        help="The number of training frames.",
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment."
    )
    parser.add_argument("--logdir", default="runs", help="The base logging directory.")
    parser.add_argument(
        "--save_freq", default=100, help="How often to save the model, in episodes."
    )
    parser.add_argument("--hyperparameters", default=[], nargs="*")
    args = parser.parse_args()

    # construct the environment
    env = env_constructor(args.env, device=args.device)

    # construct the agents
    agent_name = args.agent
    agent = getattr(presets, agent_name)
    agent = agent.device(args.device)

    # parse hyperparameters
    hyperparameters = {}
    for hp in args.hyperparameters:
        key, value = hp.split("=")
        hyperparameters[key] = type(agent.default_hyperparameters[key])(value)
    agent = agent.hyperparameters(**hyperparameters)

    # run the experiment
    run_experiment(
        agent,
        env,
        args.frames,
        render=args.render,
        logdir=args.logdir,
        save_freq=args.save_freq,
    )
