'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
from all.presets.classic_control import dqn, a2c
from all.environments import GymEnvironment

def main():
    device = 'cpu'
    timesteps = 40000
    run_experiment(
        [dqn(), a2c()],
        [GymEnvironment('CartPole-v0', device), GymEnvironment('Acrobot-v1', device)],
        timesteps,
    )
    plot_returns_100('runs', timesteps=timesteps)

if __name__ == "__main__":
    main()
