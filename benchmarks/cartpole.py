from all.environments import GymEnvironment
from all.presets import classic_control
from runner import run

if __name__ == '__main__':
    run(classic_control, GymEnvironment('CartPole-v0'), episodes=1000)
