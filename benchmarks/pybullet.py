from all.experiments import SlurmExperiment
from all.presets.continuous import ddpg, ppo, sac
from all.environments import PybulletEnvironment


def main():
    device = 'cuda'

    frames = int(1e7)

    agents = [
        ddpg,
        ppo,
        sac
    ]

    envs = [PybulletEnvironment(env, device) for env in PybulletEnvironment.short_names]

    SlurmExperiment(agents, envs, frames, sbatch_args={
        'partition': '1080ti-long'
    })


if __name__ == "__main__":
    main()
