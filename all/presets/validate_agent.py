from all.environments import GymEnvironment

def validate_agent(make_agent, env):
    if isinstance(env, str):
        env = GymEnvironment(env)
    agent = make_agent(env)
    # Run two episodes, enough to
    # exercise all parts of the agent
    # in most cases.
    for _ in range(2):
        env.reset()
        env.step(agent.initial(env.state))
        while not env.done:
            env.step(agent.act(env.state, env.reward))
        agent.terminal(env.reward)
