import torch
from all.environments import State
from all.logging import DummyWriter

def validate_agent(make_agent, env):
    if isinstance(make_agent, tuple):
        validate_multi_env_agent(make_agent, env)
    else:
        validate_single_env_agent(make_agent, env)

def validate_single_env_agent(make_agent, env):
    agent = make_agent(env, writer=DummyWriter())
    # Run two episodes, enough to
    # exercise all parts of the agent
    # in most cases.
    for _ in range(2):
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))
        agent.act(env.state, env.reward)

def validate_multi_env_agent(make_agent, base_env):
    make, n_env = make_agent
    envs = base_env.duplicate(n_env)
    agent = make(envs, writer=DummyWriter())

    for env in envs:
        env.reset()

    for _ in range(10):
        states = State.from_list([env.state for env in envs])
        rewards = [env.reward for env in envs]
        rewards = torch.tensor(rewards, device=base_env.device).float()
        actions = agent.act(states, rewards)
        for (action, env) in zip(actions, envs):
            if env.done:
                env.reset()
            elif action is not None:
                env.step(action)
