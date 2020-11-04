from all import nn


def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )


def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, env.action_space.n)
            ),
        )
    )


def nature_features(frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )


def nature_value_head():
    return nn.Linear(512, 1)


def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)


def nature_c51(env, frames=4, atoms=51):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n * atoms)
    )


def nature_rainbow(env, frames=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    env.action_space.n * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )
