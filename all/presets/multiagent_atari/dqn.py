import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss
from all.approximation import Approximation, FixedTarget
from all.agents import DQN, IndependentMultiagent, Multiagent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all import nn

class NatureDqnParameterSharing(nn.Module):
    def __init__(self, env, frames=4):
        super().__init__()
        n_agents = len(env.agents)
        n_actions = env.action_spaces['first_0'].n
        self.conv = nn.Sequential(
            nn.Scale(1/255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.hidden = nn.Linear(3136 + n_agents, 512)
        self.output = nn.Linear0(512 + n_agents, n_actions)

    def forward(self, states, actions=None):
        observation = states.as_input('observation')
        agent = states.as_input('agent')
        # get the convolution features
        x = self.conv(observation)
        # add one-hot agent id encoding
        x = torch.cat((x, agent), dim=1)
        # hidden linear layer
        x = self.hidden(x)
        # add one-hot agent id encoding again
        x = torch.cat((x, agent), dim=1)
        # output layer
        x = self.output(x)
        # transform to correct shape and apply mask
        x = states.apply_mask(states.as_output(x))

        if actions is None:
            return x
        return x.gather(1, actions.view(-1, 1)).squeeze(1)

class MultiagentEncoder(Multiagent):
    def __init__(self, multiagent, agents, device):
        self.multiagent = multiagent
        self.encodings = {}
        for n, agent in enumerate(agents):
            encoding = torch.zeros(len(agents), device=device)
            encoding[n] = 1.
            self.encodings[agent] = encoding

    def act(self, agent, state):
        return self.multiagent.act(agent, state.update('agent', self.encodings[agent]))

    def eval(self, agent, state):
        return self.multiagent.eval(agent, state.update('agent', self.encodings[agent]))

def dqn(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
        # Model construction
        model_constructor=NatureDqnParameterSharing
):
    """
    DQN Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (int): Final probability of choosing a random action.
        final_exploration_frame (int): The frame where the exploration decay stops.
        model_constructor (function): The function used to construct the neural model.
    """
    def _dqn(env, writers=None):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency
        final_exploration_step = final_exploration_frame / action_repeat

        n_agents = len(env.agents)
        n_actions = env.action_spaces['first_0'].n

        model = model_constructor(env).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            eps=eps
        )

        q = Approximation(
            model,
            optimizer,
            scheduler=CosineAnnealingLR(optimizer, last_update),
            target=FixedTarget(target_update_frequency),
            writer=writers['first_0']
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        def agent_constructor(writer):
            policy = GreedyPolicy(
                q,
                n_actions,
                epsilon=LinearScheduler(
                    initial_exploration,
                    final_exploration,
                    replay_start_size,
                    final_exploration_step - replay_start_size,
                    name="epsilon",
                    writer=writer
                )
            )

            return DeepmindAtariBody(
                DQN(
                    q,
                    policy,
                    replay_buffer,
                    discount_factor=discount_factor,
                    loss=smooth_l1_loss,
                    minibatch_size=minibatch_size,
                    replay_start_size=replay_start_size,
                    update_frequency=update_frequency,
                ),
                lazy_frames=True
            )

        return MultiagentEncoder(IndependentMultiagent({
            agent : agent_constructor(writers[agent])
            for agent in env.agents
        }), env.agents, device)
    return _dqn
