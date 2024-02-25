from all.environments import GymEnvironment
from all.presets import classic_control

from .train import train


def main():
    train(
        classic_control,
        GymEnvironment,
        description="Train an agent on an classic control environment.",
        env_help="The name of the environment (e.g., CartPole-v0).",
        default_frames=50000,
    )


if __name__ == "__main__":
    main()
