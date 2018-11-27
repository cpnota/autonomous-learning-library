from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import LearningCurve

def run():
    learning_curve = LearningCurve('FrozenLake-v0', trials=1000)
    learning_curve.run(sarsa, plot_every=1000, print_every=1000)
    learning_curve.plot()

if __name__ == '__main__':
    run()
