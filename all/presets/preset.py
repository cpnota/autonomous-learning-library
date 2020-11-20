from abc import ABC, abstractmethod
import torch


class Preset(ABC):
    """
    A Preset agent configuration.

    Args:
        n_envs (int, optional): If the Preset is for a ParallelAgent,
            sets the number of parallel environments.

    Attributes:
        n_envs: If the Preset is for a ParallelAgent, the number of parallel environments.
    """
    def __init__(self, n_envs=None):
        self.n_envs = n_envs


    @abstractmethod
    def agent(self, writer=None, train_steps=float('inf')):
        """
        Instansiate a training-mode agent with the existing model.

        Args:
            writer (all.logging.Writer, optional): Coefficient for the entropy term in the total loss.
            train_steps (int, optional): The number of steps for which the agent will be trained.

        Returns:
            all.agents.Agent: The instansiated Agent.
        """
        pass

    @abstractmethod
    def test_agent(self):
        """
        Instansiate a test-mode agent with the existing model.

        Returns:
            all.agents.Agent: The instansiated test Agent.
        """
        pass


    def save(self, filename):
        """
        Save the preset and the contained model to disk.

        The preset can later be loaded using torch.load(filename), allowing
        a test mode agent to be instansiated for evaluation or other purposes.
        
        Args:
            filename (str): The path where the preset should be saved.
        """
        return torch.save(self, filename)


    def is_parallel(self):
        """
        Determine whether the Preset is for a parallel Agent.

        Returns:
            bool: True if the Preset is for a parallel Agent, False otherwise.
        """
        return self.n_envs is not None
