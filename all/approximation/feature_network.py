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
        self._cache = []
        self._out = []

    def __call__(self, states):
        '''
        Run a forward pass of the model and return the detached output.

        Args:
            state (all.environment.State): An environment State

        Returns:
            all.environment.State: An environment State with the computed features
        '''
        features = self.model(states)
        graphs = features.observation
        observation = graphs.detach()
        observation.requires_grad = True
        features['observation'] = observation
        self._enqueue(graphs, observation)
        return features

    def reinforce(self):
        '''
        Backward pass of the model.
        '''
        graphs, grads = self._dequeue()
        if graphs.requires_grad:
            graphs.backward(grads)
            self.step()

    def _enqueue(self, features, out):
        self._cache.append(features)
        self._out.append(out)

    def _dequeue(self):
        graphs = []
        grads = []
        for graph, out in zip(self._cache, self._out):
            if out.grad is not None:
                graphs.append(graph)
                grads.append(out.grad)
        self._cache = []
        self._out = []
        return torch.cat(graphs), torch.cat(grads)


class FeatureModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = states.as_output(self.model(states.as_input('observation')))
        return states.update('observation', features)
