import copy
import torch
from torch.nn import utils
from torch.nn.functional import mse_loss
from all.experiments import DummyWriter

class Approximation():
    def __init__(
            self,
            model,
            optimizer,
            clip_grad=0,
            loss_scaling=1,
            loss=mse_loss,
            name='approximation',
            target_update_frequency=None,
            writer=DummyWriter(),
    ):
        self.model = model
        self.target_model = self._init_target_model(target_update_frequency)
        self.device = next(model.parameters()).device
        self._updates = 0
        self._target_update_frequency = target_update_frequency
        self._optimizer = optimizer
        self._loss = loss
        self._loss_scaling = loss_scaling
        self._cache = []
        self._clip_grad = clip_grad
        self._writer = writer
        self._name = name

    def __call__(self, *inputs):
        result = self.model(*inputs)
        self._enqueue(result)
        return result.detach()

    def eval(self, *inputs):
        with torch.no_grad():
            training = self.target_model.training
            result = self.target_model(*inputs)
            self.target_model.train(training)
            return result

    def reinforce(self, errors, retain_graph=False):
        batch_size = len(errors)
        cache = self._dequeue(batch_size)
        if cache.requires_grad:
            loss = self._loss(cache, errors) * self._loss_scaling
            self._writer.add_loss(self._name, loss)
            loss.backward(retain_graph=retain_graph)
            self._step()

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

    def _step(self):
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._update_target_model()

    def _init_target_model(self, target_update_frequency):
        return (
            copy.deepcopy(self.model)
            if target_update_frequency is not None
            else self.model
        )

    def _update_target_model(self):
        self._updates += 1
        if self._should_update_target():
            self.target_model.load_state_dict(self.model.state_dict())

    def _should_update_target(self):
        return (
            (self._target_update_frequency is not None)
            and (self._updates % self._target_update_frequency == 0)
        )
