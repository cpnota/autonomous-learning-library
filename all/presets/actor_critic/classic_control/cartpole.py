from all.experiments import Experiment
from all.presets.actor_critic import ac_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=1000, trials=1)
    experiment.run(
        ac_cc(),
        label="actor-critic",
        render=True
    )

if __name__ == '__main__':
    run()
