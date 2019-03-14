from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.rainbow import rainbow

def run():
    # env = PongEnvironment()
    env = AtariEnvironment("Pong")
    experiment = Experiment(env, episodes=250, trials=1)
    experiment.run(
        rainbow(
            # parameters from:
            # https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55
            replay_buffer_size=100000,
            replay_start_size=10000,
            target_update_frequency=1000,
            update_frequency=2,
            minibatch_size=2 * 32,
            # hand tuned
            final_beta_frame=2e6 # approximate number of training frames
        ),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="rainbow-pong.png")
    experiment.save("rainbow_pong")


if __name__ == '__main__':
    run()
