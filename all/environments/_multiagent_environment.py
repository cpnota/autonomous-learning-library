from abc import ABC, abstractmethod


class MultiagentEnvironment(ABC):
    '''
    A multiagent reinforcement learning Environment.

    The Multiagent variant of the Environment object.
    An Environment defines the dynamics of a particular problem:
    the states, the actions, the transitions between states, and the rewards given to the agent.
    Environments are often used to benchmark reinforcement learning agents,
    or to define real problems that the user hopes to solve using reinforcement learning.
    '''

    @abstractmethod
    def reset(self):
        '''
        Reset the environment and return a new initial state for the first agent.

        Returns
            all.core.MultiagentState: The initial state for the next episode.
        '''

    @abstractmethod
    def step(self, action):
        '''
        Apply an action for the current agent and get the multiagent state for the next agent.

        Parameters:
            action: The Action for the current agent and timestep.

        Returns:
            all.core.MultiagentState: The state for the next agent.
        '''

    @abstractmethod
    def render(self, **kwargs):
        '''Render the current environment state.'''

    @abstractmethod
    def close(self):
        '''Clean up any extraneous environment objects.'''

    @abstractmethod
    def agent_iter(self):
        '''
        Create an iterable which that the next element is always the name of the agent whose turn it is to act.

        Returns:
            An Iterable over Agent strings.
        '''

    @abstractmethod
    def last(self):
        '''
        Get the MultiagentState object for the current agent.

        Returns:
            The all.core.MultiagentState object for the current agent.
        '''

    @abstractmethod
    def is_done(self, agent):
        '''
        Determine whether a given agent is done.

        Args:
            agent (str): The name of the agent.

        Returns:
            A boolean representing whether the given agent is done.
        '''

    @property
    def state(self):
        '''The State for the current agent.'''
        return self.last()

    @property
    @abstractmethod
    def name(self):
        '''str: The name of the environment.'''

    @property
    @abstractmethod
    def state_spaces(self):
        '''A dictionary of state spaces for each agent.'''

    @property
    def observation_spaces(self):
        '''Alias for MultiagentEnvironment.state_spaces.'''
        return self.state_space

    @property
    @abstractmethod
    def action_spaces(self):
        '''A dictionary of action spaces for each agent.'''

    @property
    @abstractmethod
    def device(self):
        '''
        The torch device the environment lives on.
        '''
