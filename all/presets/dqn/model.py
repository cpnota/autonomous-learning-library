from torch import nn

# Not provided by Pytorch,
# because devs = nazis
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def deep_q_atari(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 16, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2816, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )
 
def deep_q_classic_control(env, frames=4):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )
