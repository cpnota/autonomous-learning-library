from all.presets.tabular import sarsa
from all.experiments import LearningCurve

def run():
    learning_curve = LearningCurve()
    learning_curve.run(sarsa, 'FrozenLake-v0')
    learning_curve.plot()

if __name__ == '__main__':
    run()
