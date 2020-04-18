import pybullet
import pybullet_envs
from all.experiments import SlurmExperiment
from all.presets.continuous import ddpg, ppo, sac
from all.environments import GymEnvironment

def main():
    device = 'cuda'

    frames = int(1e7)

    agents = [
        ddpg(last_frame=frames),
        ppo(last_frame=frames),
        sac(last_frame=frames)
    ]

    envs = [GymEnvironment(env, device) for env in [
        'AntBulletEnv-v0',
        "HalfCheetahBulletEnv-v0",
        'HumanoidBulletEnv-v0',
        'HopperBulletEnv-v0',
        'Walker2DBulletEnv-v0'
    ]]

    SlurmExperiment(agents, envs, frames, sbatch_args={
        'partition': '1080ti-long'
    })

if __name__ == "__main__":
    main()
