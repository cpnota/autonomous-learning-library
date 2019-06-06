import torch
from torch.nn import utils
from all.environments import State
from .features import Features

class FeatureNetwork(Features):
    def __init__(self, model, optimizer, clip_grad=0):
        self.model = model
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self._cache = []
        self._out = []

    def __call__(self, states):
        features = self.model(states.features.float())
        out = features.detach()
        out.requires_grad = True
        self._cache.append(features)
        self._out.append(out)
        return State(
            out,
            mask=states.mask,
            info=states.info
        )

    def eval(self, states):
        with torch.no_grad():
            training = self.model.training
            result = self.model(states.features.float())
            self.model.train(training)
            return State(
                result,
                mask=states.mask,
                info=states.info
            )

    def reinforce(self):
        graphs, grads = self._decache()
        graphs.backward(grads)
        if self.clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _decache(self):
        graphs = []
        grads = []

        new_cache = []
        new_out = []
        if self._out[-1].grad is None:
            new_cache = [self._cache[-1]]
            new_out = [self._out[-1]]
            self._cache = self._cache[0:-1]
            self._out = self._out[0:-1]

        for graph, out in zip(self._cache, self._out):
            if out.grad is not None:
                graphs.append(graph)
                grads.append(out.grad)
        graphs = torch.cat(graphs)
        grads = torch.cat(grads)
        self._cache = new_cache
        self._out = new_out
        return graphs, grads
