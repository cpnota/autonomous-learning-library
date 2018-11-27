from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import LearningCurve

def run():
    learning_curve = LearningCurve('FrozenLake-v0')
    learning_curve.run(sarsa)
    learning_curve.run(actor_critic)
    learning_curve.plot()

if __name__ == '__main__':
    run()
