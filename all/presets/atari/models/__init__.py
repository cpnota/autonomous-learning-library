from all import nn

def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )

def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear0(512, env.action_space.n)
            ),
        )
    )

def nature_cnn():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
    )

def nature_value_head():
    return nn.Linear0(512, 1)

def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)
