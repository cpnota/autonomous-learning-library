from all.experiments import SlurmExperiment
from all.presets.continuous import ddpg, ppo, sac
from all.environments import PybulletEnvironment


def main():
    frames = int(1e7)

    agents = [
        ddpg,
        ppo,
        sac
    ]

    envs = [PybulletEnvironment(env, device='cuda') for env in PybulletEnvironment.short_names]

    SlurmExperiment(agents, envs, frames, logdir='benchmarks/pybullet', sbatch_args={
        'partition': 'gpu-long'
    })


if __name__ == "__main__":
    main()
