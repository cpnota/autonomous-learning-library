from all.environments import PybulletEnvironment
from all.presets import continuous

from .train import train


def main():
    train(
        continuous,
        PybulletEnvironment,
        description="Train an agent on an PyBullet environment.",
        env_help="The name of the environment (e.g., AntBulletEnv-v0).",
        default_frames=10e6,
    )


if __name__ == "__main__":
    main()
