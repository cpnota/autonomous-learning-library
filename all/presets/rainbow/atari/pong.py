from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.rainbow import rainbow

def run():
    # env = PongEnvironment()
    env = AtariEnvironment("Pong")
    experiment = Experiment(env, episodes=500, trials=1)
    experiment.run(
        rainbow(
            # parameters from:
            # https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55
            replay_buffer_size=100000,
            replay_start_size=10000,
            initial_exploration=1.00,
            final_exploration=0.02,
            final_exploration_frame=400000,
            target_update_frequency=1000,
            update_frequency=2,
            minibatch_size=2 * 32,
            # Prioritized Replay
            alpha=0.6,
            beta=0.5,
            final_beta_frame=400000,
        ),
        render=True
    )


if __name__ == '__main__':
    run()
