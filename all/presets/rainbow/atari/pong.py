from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.rainbow import rainbow

def run():
    # env = PongEnvironment()
    env = AtariEnvironment("Pong")
    experiment = Experiment(env, episodes=500, trials=1)
    experiment.run(
        rainbow(
            # Vanilla DQN
            minibatch_size=4 * 32,
            replay_buffer_size=250000, # originally 1e6
            discount_factor=0.99,
            update_frequency=4,
            lr=1e-4,
            replay_start_size=2e4,
            # Double Q-Learning
            target_update_frequency=1000,
            # Prioritized Replay
            alpha=0.5,
            beta=0.4,
            final_beta_frame=200e6,
            # NoisyNets
            sigma_init=0.5
        ),
        plot_every=5,
        print_every=1,
        render=True
    )
    experiment.plot(filename="rainbow-pong.png", frequency=5)
    experiment.save("rainbow_pong")


if __name__ == '__main__':
    run()
