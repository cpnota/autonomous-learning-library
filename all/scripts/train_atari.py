from all.environments import AtariEnvironment
from all.presets import atari

from .train import train


def main():
    train(
        atari,
        AtariEnvironment,
        description="Train an agent on an Atari environment.",
        env_help="The name of the environment (e.g., 'Pong').",
        default_frames=40e6,
    )


if __name__ == "__main__":
    main()
