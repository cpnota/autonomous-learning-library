import torch
from torch.nn import utils
from torch.nn.functional import mse_loss
from all.experiments import DummyWriter
from .target import TrivialTarget

class Approximation():
    def __init__(
            self,
            model,
            optimizer,
            clip_grad=0,
            loss_scaling=1,
            loss=mse_loss,
            name='approximation',
            target=None,
            writer=DummyWriter(),
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self._target = target or TrivialTarget()
        self._target.init(model)
        self._updates = 0
        self._optimizer = optimizer
        self._loss = loss
        self._loss_scaling = loss_scaling
        self._cache = []
        self._clip_grad = clip_grad
        self._writer = writer
        self._name = name

    def __call__(self, *inputs, detach=True):
        result = self.model(*inputs)
        if detach:
            self._enqueue(result)
            return result.detach()
        return result

    def eval(self, *inputs):
        return self._target(*inputs)

    def reinforce(self, errors, retain_graph=False):
        batch_size = len(errors)
        cache = self._dequeue(batch_size)
        if cache.requires_grad:
            loss = self._loss(cache, errors) * self._loss_scaling
            self._writer.add_loss(self._name, loss)
            loss.backward(retain_graph=retain_graph)
            self.step()

    def step(self):
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._target.update()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _enqueue(self, results):
        self._cache.append(results)

    def _dequeue(self, batch_size):
        i = 0
        num_items = 0
        while num_items < batch_size and i < len(self._cache):
            num_items += len(self._cache[i])
            i += 1
        if num_items != batch_size:
            raise ValueError("Incompatible batch size.")
        items = torch.cat(self._cache[:i])
        self._cache = self._cache[i:]
        return items
