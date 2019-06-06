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

    def __call__(self, states):
        features = self.model(states.features)
        out = features.detach()
        out.requires_grad = True
        self._cache.append(features)
        return State(
            out,
            mask=states.mask,
            info=states.info
        )

    def eval(self, states):
        with torch.no_grad():
            training = self.model.training
            result = self.model(states.features)
            self.model.train(training)
            return State(
                result,
                mask=states.mask,
                info=states.info
            )

    def reinforce(self, grad):
        batch_size = len(grad)
        cache = self._decache(batch_size)

        if cache.requires_grad:
            cache.backward(grad)
            if self.clip_grad != 0:
                utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _decache(self, batch_size):
        i = 0
        items = 0
        while items < batch_size and i < len(self._cache):
            items += len(self._cache[i])
            i += 1
        if items != batch_size:
            raise ValueError("Incompatible batch size.")

        cache = torch.cat(self._cache[:i])
        self._cache = self._cache[i:]
        return cache
