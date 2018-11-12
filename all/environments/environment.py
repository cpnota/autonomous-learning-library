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

    @abstractmethod
    def reset(self):
        """
        Reset the environment and return a new intial state.

        Returns
        -------
        State
            The initial state for the next episode.
        """
        pass

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
        State
            The state of the environment after the action is applied
        float
            The reward achieved by the previous action
        done
            True if the environment has entered a terminal state and should be reset
        info
            Diagnostic information useful for debugging
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the current environment state.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up any extraneaous environment objects.
        """
        pass

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
        pass

    @property
    def observation_space(self):
        """
        Alias for Environemnt.state_space.

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
        pass

    @property
    @abstractmethod
    def state(self):
        """
        The State of the Environment at the current timestep.
        """
        pass

    @property
    @abstractmethod
    def action(self):
        """
        The most recent Action taken
        """
        pass

    @property
    @abstractmethod
    def reward(self):
        """
        The reward for the previous action taken
        """
        pass

    @property
    @abstractmethod
    def done(self):
        """
        Whether or not the environment has terminated and should be reset.
        """
        pass

    @property
    def info(self):
        """
        Debugging info for the current time step.
        """
        return None
