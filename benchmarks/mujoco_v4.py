from all.environments import MujocoEnvironment
from all.experiments import SlurmExperiment
from all.presets.continuous import ddpg, ppo, sac


def main():
    frames = int(5e6)

    agents = [ddpg, ppo, sac]

    envs = [
        MujocoEnvironment(env, device="cuda")
        for env in [
            "Ant-v4",
            "HalfCheetah-v4",
            "Hopper-v4",
            "Humanoid-v4",
            "Walker2d-v4",
        ]
    ]

    SlurmExperiment(
        agents,
        envs,
        frames,
        logdir="benchmarks/mujoco_v4",
        sbatch_args={
            "partition": "gpu-long",
        },
    )


if __name__ == "__main__":
    main()
