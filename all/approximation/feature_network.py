import torch
from all.environments import State
from .approximation import Approximation

class FeatureNetwork(Approximation):
    def __init__(self, model, optimizer=None, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self._cache = []
        self._out = []

    def __call__(self, states):
        features = self.model(states.features.float())
        out = features.detach()
        out.requires_grad = True
        self._enqueue(features, out)
        return State(
            out,
            mask=states.mask,
            info=states.info
        )

    def eval(self, states):
        with torch.no_grad():
            training = self.target_model.training
            result = self.target_model(states.features.float())
            self.target_model.train(training)
            return State(
                result,
                mask=states.mask,
                info=states.info
            )

    def reinforce(self):
        graphs, grads = self._dequeue()
        graphs.backward(grads)
        self._step()

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
