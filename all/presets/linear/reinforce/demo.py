from all.environments import GymWrapper
from all.presets.linear import reinforce

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
    env = GymWrapper('CartPole-v0')
    # pylint: disable=protected-access
    agent = reinforce(env)

    for _ in range(1):
        for _ in range(2000):
            run_episode(agent, env)

    env.close()


if __name__ == '__main__':
    run()
