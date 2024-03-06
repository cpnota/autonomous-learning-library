from all.environments import PybulletEnvironment
from all.experiments import SlurmExperiment
from all.presets.continuous import ddpg, ppo, sac


def main():
    frames = int(5e6)

    agents = [ddpg, ppo, sac]

    envs = [
        PybulletEnvironment(env, device="cuda")
        for env in [
            "AntBulletEnv-v0",
            "HalfCheetahBulletEnv-v0",
            "HopperBulletEnv-v0",
            "HumanoidBulletEnv-v0",
            "Walker2DBulletEnv-v0",
        ]
    ]

    SlurmExperiment(
        agents,
        envs,
        frames,
        logdir="benchmarks/pybullet",
        sbatch_args={
            "partition": "gpu-long",
        },
    )


if __name__ == "__main__":
    main()
