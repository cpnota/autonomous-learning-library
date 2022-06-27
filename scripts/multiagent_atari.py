import argparse
from all.environments import MultiagentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets import IndependentMultiagentPreset


class DummyEnv():
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space


def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. pong_v2).")
    parser.add_argument(
        "agents", nargs='*', help="List of agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--replay_buffer_size",
        default=100000,
        help="The size of the replay buffer, if applicable",
    )
    parser.add_argument(
        "--frames", type=int, default=40e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment."
    )
    parser.add_argument(
        "--logger", default='tensorboard', help="The backend used for tracking experiment metrics."
    )
    args = parser.parse_args()

    env = MultiagentAtariEnv(args.env, device=args.device)

    presets = {
        agent_id: getattr(atari, agent_type)
            .hyperparameters(replay_buffer_size=args.replay_buffer_size)
            .device(args.device)
            .env(env.subenvs[agent_id])
            .build()
        for agent_id, agent_type in zip(env.agents, args.agents)
    }

    experiment = MultiagentEnvExperiment(
        IndependentMultiagentPreset('Independent', args.device, presets),
        env,
        verbose=False,
        render=args.render,
        logger=args.logger,
    )
    experiment.train(frames=args.frames)
    experiment.save()
    experiment.test(episodes=100)
    experiment.close()


if __name__ == "__main__":
    main()
