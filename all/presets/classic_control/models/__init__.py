from all import nn

def fc_relu_q(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        nn.Linear(hidden, env.action_space.n)
    )

def dueling_fc_relu_q(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n)
            )
        )
    )

def fc_relu_features(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
    )

def fc_value_head(hidden=64):
    return nn.Linear0(hidden, 1)

def fc_policy_head(env, hidden=64):
    return nn.Linear0(hidden, env.action_space.n)
