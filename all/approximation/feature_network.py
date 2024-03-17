import torch

from .approximation import Approximation


class FeatureNetwork(Approximation):
    """
    An Approximation that accepts a state updates the observation key
    based on the given model.
    """

    def __init__(self, model, optimizer=None, name="feature", **kwargs):
        model = FeatureModule(model)
        super().__init__(model, optimizer, name=name, **kwargs)


class FeatureModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = states.as_output(self.model(states.as_input("observation")))
        return states.update("observation", features)
