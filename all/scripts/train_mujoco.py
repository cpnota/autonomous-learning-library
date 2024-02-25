from all.environments import MujocoEnvironment
from all.presets import continuous

from .train import train


def main():
    train(
        continuous,
        MujocoEnvironment,
        description="Train an agent on an Mujoco environment.",
        env_help="The name of the environment (e.g., Ant-v4).",
        default_frames=10e6,
    )


if __name__ == "__main__":
    main()
