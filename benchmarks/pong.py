from all.environments import AtariEnvironment
from all.presets import atari
from runner import run

if __name__ == '__main__':
    run(atari, AtariEnvironment('Pong'), frames=20e6)
