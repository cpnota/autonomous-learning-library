from all.experiments import Experiment
from all.environments import GymEnvironment
from all.presets.dqn import dqn

def run():
    env = GymEnvironment("BreakoutNoFrameskip-v4")
    experiment = Experiment(env, episodes=40000, trials=1)
    experiment.run(
        dqn(),
        plot_every=100,
        print_every=1,
        render=True
    )
    experiment.plot(filename="dqn-breakout-dueling.png", frequency=100)
    experiment.save("dqn_breakout-dueling")


if __name__ == '__main__':
    run()
