from abc import ABC, abstractmethod


class Environment(ABC):
    """
    A reinforcement learning Environment.

    In reinforcement learning, an Agent learns by interacting with an Environment.
    An Environment defines the dynamics of a particular problem:
    the states, the actions, the transitions between states, and the rewards given to the agent.
    Environments are often used to benchmark reinforcement learning agents,
    or to define real problems that the user hopes to solve using reinforcement learning.
    """

    @property
    @abstractmethod
    def name(self):
        """
        The name of the environment.
        """

    @abstractmethod
    def reset(self):
        """
        Reset the environment and return a new initial state.

        Returns
        -------
        State
            The initial state for the next episode.
        """

    @abstractmethod
    def step(self, action):
        """
        Apply an action and get the next state.

        Parameters
        ----------
        action : Action
            The action to apply at the current time step.

        Returns
        -------
        all.environments.State
            The State of the environment after the action is applied.
            This State object includes both the done flag and any additional "info"
        float
            The reward achieved by the previous action
        """

    @abstractmethod
    def render(self, **kwargs):
        """
        Render the current environment state.
        """

    @abstractmethod
    def close(self):
        """
        Clean up any extraneous environment objects.
        """

    @property
    @abstractmethod
    def state(self):
        """
        The State of the Environment at the current timestep.
        """

    @property
    @abstractmethod
    def state_space(self):
        """
        The Space representing the range of observable states.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """

    @property
    def observation_space(self):
        """
        Alias for Environment.state_space.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        return self.state_space

    @property
    @abstractmethod
    def action_space(self):
        """
        The Space representing the range of possible actions.

        Returns
        -------
        Space
            An object of type Space that represents possible actions the agent may take
        """

    @abstractmethod
    def duplicate(self, n):
        """
        Create n copies of this environment.
        """

    @property
    @abstractmethod
    def device(self):
        """
        The torch device the environment lives on.
        """
