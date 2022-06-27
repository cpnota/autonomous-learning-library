from all.experiments import SlurmExperiment
from all.presets import atari
from all.environments import AtariEnvironment


def main():
    agents = [
        atari.a2c,
        atari.c51,
        atari.dqn,
        atari.ddqn,
        atari.ppo,
        atari.rainbow,
    ]
    envs = [AtariEnvironment(env, device='cuda') for env in ['BeamRider', 'Breakout', 'Pong', 'Qbert', 'SpaceInvaders']]
    SlurmExperiment(agents, envs, 10e6, logdir='benchmarks/atari40', sbatch_args={
        'partition': 'gpu-long'
    })


if __name__ == "__main__":
    main()
