import torch
from all.environments import State
from .approximation import Approximation


class FeatureNetwork(Approximation):
    def __init__(self, model, optimizer=None, name='feature', **kwargs):
        model = FeatureModule(model)
        super().__init__(model, optimizer, name=name, **kwargs)
        self._cache = []
        self._out = []

    def __call__(self, states):
        features = self.model(states)
        graphs = features.raw
        # pylint: disable=protected-access
        features._raw = graphs.detach()
        features._raw.requires_grad = True
        self._enqueue(graphs, features._raw)
        return features

    def reinforce(self):
        graphs, grads = self._dequeue()
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
        features = self.model(states.features.float())
        return State(
            features,
            mask=states.mask,
            info=states.info
        )
