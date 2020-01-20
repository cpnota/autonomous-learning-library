from abc import ABC, abstractmethod
from all.optim import Schedulable

class Agent(ABC, Schedulable):
    """
    A reinforcement learning agent.

    In reinforcement learning, an Agent learns by interacting with an Environment.
    Usually, an agent tries to maximize a reward signal.
    It does this by observing environment "states", taking "actions", receiving "rewards",
    and in doing so, learning which state-action pairs correlate with high rewards.
    An Agent implementation should encapsulate some particular reinforcement learning algorihthm.
    """

    @abstractmethod
    def act(self, state, reward):
        """
        Select an action for the current timestep and update internal parameters.

        In general, a reinforcement learning agent does several things during a timestep:
        1. Choose an action,
        2. Compute the TD error from the previous time step
        3. Update the value function and/or policy
        The order of these steps differs depending on the agent.
        This method allows the agent to do whatever is necessary for itself on a given timestep.
        However, the agent must ultimately return an action.

        Args:
            state (all.environment.State): The environment state at the current timestep.
            reward (torch.Tensor): The reward from the previous timestep.
            info (:obj:, optional): The info object from the environment.

        Returns:
            torch.Tensor: The action to take at the current timestep.
        """
