from all.environments import make_atari
from all.presets.dqn import sarsa

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
    env = make_atari('BreakoutDeterministic-v4')
    # pylint: disable=protected-access
    agent = sarsa(env)

    for _ in range(1):
        for _ in range(200):
            run_episode(agent, env)

    env.close()


if __name__ == '__main__':
    run()
