from .c51 import C51, C51TestAgent


class Rainbow(C51):
    """
    Rainbow: Combining Improvements in Deep Reinforcement Learning.
    Rainbow combines C51 with 5 other "enhancements" to
    DQN: double Q-learning, dueling networks, noisy networks
    prioritized reply, n-step rollouts.
    https://arxiv.org/abs/1710.02298

    Whether this agent is Rainbow or C51 depends
    on the objects that are passed into it.
    Dueling networks and noisy networks are part
    of the model used for q_dist, while
    prioritized replay and n-step rollouts are handled
    by the replay buffer.
    Double Q-learning is always used.

    Args:
        q_dist (QDist): Approximation of the Q distribution.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        eps (float): Stability parameter for computing the loss function.
        exploration (float): The probability of choosing a random action.
        minibatch_size (int): The number of experiences to sample in
            each training update.
        replay_start_size (int): Number of experiences in replay buffer
            when training begins.
        update_frequency (int): Number of timesteps per training update.
    """


RainbowTestAgent = C51TestAgent
