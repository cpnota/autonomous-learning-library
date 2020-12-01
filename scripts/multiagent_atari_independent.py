import argparse
from all.environments import MultiAgentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets.multiagent_atari import IndependentMultiagentAtariPreset


def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
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
    parser.add_argument('--hyperparameters', default=[], nargs='*')
    args = parser.parse_args()

    agent_name = args.agent
    agent = getattr(atari, agent_name)().device(args.device)
    env = MultiAgentAtariEnv(args.env, device=args.device)

    # parse hyperparameters
    hyperparameters = {}
    for hp in args.hyperparameters:
        key, value = hp.split('=')
        hyperparameters[key] = type(agent._hyperparameters[key])(value)
    agent = agent.hyperparameters(**hyperparameters)

    experiment = MultiagentEnvExperiment(
        IndependentMultiagentAtariPreset(env, agent),
        env,
        write_loss=False
    )
    experiment.train()


if __name__ == "__main__":
    main()
