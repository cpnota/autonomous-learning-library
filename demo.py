from all.environments import GymWrapper
from all.presets.fourier import Sarsa


def run_episode(agent, env):
    env.reset()
    agent.new_episode(env)
    returns = 0

    while not env.done:
        env.env.render()
        agent.act()
        returns += env.reward

    print('Returns: ', returns)


def run():
    env = GymWrapper('MountainCar-v0')
    env.env._max_episode_steps = 3000  # defaults to 200
    agent = Sarsa(env)

    for _ in range(1):
        for _ in range(200):
            run_episode(agent, env)

    env.close()


if __name__ == '__main__':
    run()
