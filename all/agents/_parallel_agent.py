from abc import ABC, abstractmethod
from all.optim import Schedulable


class ParallelAgent(ABC, Schedulable):
    """
    A reinforcement learning agent that chooses actions for multiple states simultaneously.
    Differs from SingleAgent in that it accepts a StateArray instead of a State to process
    input from multiple environments in parallel.

    In reinforcement learning, an Agent learns by interacting with an Environment.
    Usually, an Agent tries to maximize a reward signal.
    It does this by observing environment "states", taking "actions", receiving "rewards",
    and learning which state-action pairs correlate with high rewards.
    An Agent implementation should encapsulate some particular reinforcement learning algorithm.
    """

    @abstractmethod
    def act(self, state_array):
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
            state_array (all.environment.StateArray): An array of states for each parallel environment.

        Returns:
            torch.Tensor: The actions to take for each parallel environmets.
        """
