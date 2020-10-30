import argparse
from all.environments import MultiAgentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets.multiagent_atari import independent

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
    args = parser.parse_args()

    env = MultiAgentAtariEnv(args.env, device=args.device)
    agent_name = args.agent
    agent = getattr(atari, agent_name)
    experiment = MultiagentEnvExperiment(independent(agent(device=args.device, replay_buffer_size=500000)), env, write_loss=False)
    experiment.train()

if __name__ == "__main__":
    main()
