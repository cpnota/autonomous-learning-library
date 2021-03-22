'''
Quick example of a2c running on slurm, a distributed cluster.
Note that it only runs for 1 million frames.
For real experiments, you will surely need a modified version of this script.
'''
from all.experiments import SlurmExperiment
from all.presets.atari import a2c, dqn
from all.environments import AtariEnvironment


def main():
    device = 'cuda'
    envs = [AtariEnvironment(env, device) for env in ['Pong', 'Breakout', 'SpaceInvaders']]
    SlurmExperiment([a2c.device(device), dqn.device(device)], envs, 1e6, sbatch_args={
        'partition': '1080ti-short'
    })


if __name__ == "__main__":
    main()
