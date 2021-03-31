from abc import ABC, abstractmethod
import torch


class Preset(ABC):
    """
    A Preset Agent factory.

    This class allows the user to instantiate preconfigured Agents and test Agents.
    All Agents constructed by the Preset share a network model and parameters.
    However, other objects, such as ReplayBuffers, are independently created for each Agent.
    The Preset can be saved and loaded from disk.
    """

    def __init__(self, name, device, hyperparameters):
        self.name = name
        self.device = device
        self.hyperparameters = hyperparameters

    @abstractmethod
    def agent(self, writer=None, train_steps=float('inf')):
        """
        Instantiate a training-mode Agent with the existing model.

        Args:
            writer (all.logging.Writer, optional): Coefficient for the entropy term in the total loss.
            train_steps (int, optional): The number of steps for which the agent will be trained.

        Returns:
            all.agents.Agent: The instantiated Agent.
        """
        pass

    @abstractmethod
    def test_agent(self):
        """
        Instansiate a test-mode Agent with the existing model.

        Returns:
            all.agents.Agent: The instantiated test Agent.
        """
        pass

    def save(self, filename):
        """
        Save the preset and the contained model to disk.

        The preset can later be loaded using torch.load(filename), allowing
        a test mode agent to be instantiated for evaluation or other purposes.

        Args:
            filename (str): The path where the preset should be saved.
        """
        return torch.save(self, filename)


class ParallelPreset(ABC):
    """
    A Preset ParallelAgent factory.

    This is the ParallelAgent version of all.presets.Preset.
    This class allows the user to instantiate preconfigured ParallelAgents and test Agents.
    All Agents constructed by the ParallelPreset share a network model and parameters.
    However, other objects, such as ReplayBuffers, are independently created for each Agent.
    The ParallelPreset can be saved and loaded from disk.
    """

    def __init__(self, name, device, hyperparameters):
        self.name = name
        self.device = device
        self.hyperparameters = hyperparameters

    @abstractmethod
    def agent(self, writer=None, train_steps=float('inf')):
        """
        Instantiate a training-mode ParallelAgent with the existing model.

        Args:
            writer (all.logging.Writer, optional): Coefficient for the entropy term in the total loss.
            train_steps (int, optional): The number of steps for which the agent will be trained.

        Returns:
            all.agents.ParallelAgent: The instantiated Agent.
        """
        pass

    @abstractmethod
    def test_agent(self):
        """
        Instantiate a test-mode Agent with the existing model.
        See also: ParallelPreset.parallel_test_agent()

        Returns:
            all.agents.Agent: The instantiated test Agent.
        """
        pass

    @abstractmethod
    def parallel_test_agent(self):
        """
        Instantiate a test-mode ParallelAgent with the existing model.
        See also: ParallelPreset.test_agent()

        Returns:
            all.agents.ParallelAgent: The instantiated test ParallelAgent.
        """
        pass

    @property
    def n_envs(self):
        return self.hyperparameters['n_envs']

    def save(self, filename):
        """
        Save the preset and the contained model to disk.

        The preset can later be loaded using torch.load(filename), allowing
        a test mode agent to be instantiated for evaluation or other purposes.

        Args:
            filename (str): The path where the preset should be saved.
        """
        return torch.save(self, filename)
