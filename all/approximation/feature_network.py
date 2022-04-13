import torch
from .approximation import Approximation


class FeatureNetwork(Approximation):
    '''
    A special type of Approximation that accumulates gradients before backpropagating them.
    This is useful when features are shared between network heads.

    The __call__ function caches the computation graph and detaches the output.
    Then, various functions approximators may backpropagate to the output.
    The reinforce() function will then backpropagate the accumulated gradients on the output
    through the original computation graph.
    '''

    def __init__(self, model, optimizer=None, name='feature', **kwargs):
        model = FeatureModule(model)
        super().__init__(model, optimizer, name=name, **kwargs)


class FeatureModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = states.as_output(self.model(states.as_input('observation')))
        return states.update('observation', features)
