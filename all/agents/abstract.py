from abc import ABC, abstractmethod

class Agent(ABC):
    """
    A reinforcement learning agent.

    In reinforcement learning, an Agent learns by interacting with an Environment.
    Usually, an agent tries to maximize a reward signal.
    It does this by observing environment "states", taking "actions", receiving "rewards",
    and in doing so, learning which state-action pairs correlate with high rewards.
    An Agent implementation should encapsulate some particular reinforcement learning algorihthm.
    """

    @abstractmethod
    def initial(self, state, info={}):
        """
        Choose an action in the initial state of a new episode.

        Reinforcement learning problems are often broken down into sequences called "episodes".
        An episode is a self-contained sequence of states, actions, and rewards.
        A "trial" consists of multiple episodes, and represents the lifetime of an agent.
        This method is called at the beginning of an episode.

        Parameters
        ----------
        state: The initial state of the new episode
        info (optional): The info object from the environment

        Returns
        _______
        action: The action to take in the initial state
        """

    @abstractmethod
    def act(self, state, reward, info={})
        """
        Select an action for the current timestep and update internal parameters.

        In general, a reinforcement learning agent does several things during a timestep:
        1. Choose an action,
        2. Compute the TD error from the previous time step
        3. Update the value function and/or policy
        The order of these steps differs depending on the agent.
        This method allows the agent to do whatever is necessary for itself on a given timestep.
        However, the agent must ultimately return an action.

        Parameters
        ----------
        state: The environment state at the current timestep
        reward: The reward from the previous timestep
        info (optional): The info object from the environment

        Returns
        _______
        action: The action to take at the current timestep
        """

    @abstractmethod
    def terminal(self, reward, info={})
        """
        Accept the final reward of the episode and perform final updates.

        After the final action is selected, it is still necessary to
        consider the reward given on the final timestep. This method
        provides a hook where the agent can examine this reward
        and perform any necessary updates.

        Parameters
        ----------
        reward: The reward from the previous timestep
        info (optional): The info object from the environment

        Returns
        _______
        None
        """
