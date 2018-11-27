from all.environments import GymWrapper
from all.presets.tabular import actor_critic


def run_episode(agent, env):
    env.reset()
    agent.new_episode(env)
    returns = 0

    while not env.done:
        agent.act()
        returns += env.reward

    print('Returns: ', returns)

def run():
    env = GymWrapper('FrozenLake-v0')
    agent = actor_critic(env)

    for _ in range(1):
        for _ in range(2500):
            run_episode(agent, env)

    env.close()


if __name__ == '__main__':
    run()
