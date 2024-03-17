from all.environments import AtariEnvironment
from all.experiments import SlurmExperiment
from all.presets import atari


def main():
    agents = [
        atari.a2c,
        atari.c51,
        atari.dqn,
        atari.ddqn,
        atari.ppo,
        atari.rainbow,
    ]
    envs = [
        AtariEnvironment(env, device="cuda")
        for env in ["BeamRider", "Breakout", "Pong", "Qbert", "SpaceInvaders"]
    ]
    SlurmExperiment(
        agents,
        envs,
        10e6,
        logdir="benchmarks/atari_40m",
        sbatch_args={"partition": "gypsum-1080ti"},
    )


if __name__ == "__main__":
    main()
