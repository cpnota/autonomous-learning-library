import numpy as np
from all import nn

def nature_cnn(frames=4):
    model = nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )

    def init(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
    model.apply(init)
    return model

def nature_dqn(env, frames=4):
    return nn.Sequential(
        nature_cnn(frames=frames),
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )

def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nature_cnn(frames=frames),
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

def nature_features(frames=4):
    return nn.Sequential(
        nature_cnn(frames=frames),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )

def nature_value_head():
    return nn.Linear(512, 1)

def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)
