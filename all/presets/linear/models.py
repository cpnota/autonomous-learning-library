from torch import nn

def tabular_action(env):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], env.action_space.n)
    )

def continuous_state(env):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], 1)
    )
