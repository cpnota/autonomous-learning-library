from abc import ABC, abstractmethod


class VectorEnvironment(ABC):
    """
    A reinforcement learning vector Environment.

    Similar to a regular RL environment except many environments are stacked together
    in the observations, rewards, and dones, and the vector environment expects
    an action to be given for each environment in step.

    Also, since sub-environments are done at different times, you do not need to
    manually reset the environments when they are done, rather the vector environment
    automatically resets environments when they are complete.
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
    def close(self):
        """
        Clean up any extraneous environment objects.
        """

    @property
    @abstractmethod
    def state_array(self):
        """
        A StateArray of the Environments at the current timestep.
        """

    @property
    @abstractmethod
    def state_space(self):
        """
        The Space representing the range of observable states for each environment.

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
        The Space representing the range of possible actions for each environment.

        Returns
        -------
        Space
            An object of type Space that represents possible actions the agent may take
        """

    @property
    @abstractmethod
    def device(self):
        """
        The torch device the environment lives on.
        """

    @property
    @abstractmethod
    def num_envs(self):
        """
        Number of environments in vector. This is the number of actions step() expects as input
        and the number of observations, dones, etc returned by the environment.
        """
