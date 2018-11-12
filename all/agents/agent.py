from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def new_episode(self, env):
        """
        Prepare the Agent for a new episode.

        Reinforcement learning problems are often broken down into sequences called "episodes".
        An episode is a self-contained sequence of states, actions, and rewards.
        A "trial" consists of multiple episodes, and represents the lifetime of an agent.
        In between episodes, the agent usually needs to reset some internal state.
        The method allows the agent to receive the specific environment instance
        for the upcoming epsiodes, and reset any internal state.

        Parameters
        ----------
        env : Environment
            Environment object for the next episode.
        """
        pass

    @abstractmethod
    def act(self):
        """
        Choose an action, apply it to the environment, and update self.

        In general, a reinforcement learning agent needs to do several things during a timestep:
        1. Choose an action,
        2. Dispatch it to the environment (i.e. env.step(action))
        3. Compute the TD error
        4. Update the value function and/or policy
        The order of these steps differs depending on the agent.
        This method allows the agent to do whatever is necessary for itself on a given timestep.
        The most important step is to dispatch an action to the environment.

        Parameters
        ----------
        None
        """
        pass
