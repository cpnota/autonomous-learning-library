'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
from all.presets.classic_control import dqn, a2c
from all.environments import GymEnvironment


def main():
    DEVICE = 'cpu'
    # DEVICE = 'cuda' # uncomment for gpu support
    timesteps = 40000
    run_experiment(
        [
            # DQN with default hyperparameters
            dqn.device(DEVICE),
            # DQN with a custom hyperparameters and a custom name.
            dqn.device(DEVICE).hyperparameters(replay_buffer_size=100).name('dqn-small-buffer'),
            # A2C with a custom name
            a2c.device(DEVICE).name('not-dqn')
        ],
        [GymEnvironment('CartPole-v0', DEVICE), GymEnvironment('Acrobot-v1', DEVICE)],
        timesteps,
    )
    plot_returns_100('runs', timesteps=timesteps)


if __name__ == "__main__":
    main()
