from all.environments import GymEnvironment
from all.presets import continuous

from .train import train


def main():
    train(
        continuous,
        GymEnvironment,
        description="Train an agent on a continuous control environment.",
        env_help="The name of the environment (e.g., MountainCarContinuous-v0).",
        default_frames=10e6,
    )


if __name__ == "__main__":
    main()
