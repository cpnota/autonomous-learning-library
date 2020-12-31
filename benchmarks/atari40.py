from all.experiments import SlurmExperiment
from all.presets import atari
from all.environments import AtariEnvironment


def main():
    device = 'cuda'
    agents = [
        atari.a2c(device=device),
        atari.c51(device=device),
        atari.dqn(device=device),
        atari.ddqn(device=device),
        atari.ppo(device=device),
        atari.rainbow(device=device),
    ]
    envs = [AtariEnvironment(env, device=device) for env in ['BeamRider', 'Breakout', 'Pong', 'Qbert', 'SpaceInvaders']]
    SlurmExperiment(agents, envs, 10e6, sbatch_args={
        'partition': '1080ti-long'
    })


if __name__ == "__main__":
    main()
