import os
import torch
from torch.nn import utils
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from .target import TrivialTarget
from .checkpointer import PeriodicCheckpointer

DEFAULT_CHECKPOINT_FREQUENCY = 200

class Approximation():
    def __init__(
            self,
            model,
            optimizer,
            clip_grad=0,
            loss_scaling=1,
            loss=mse_loss,
            name='approximation',
            scheduler=None,
            target=None,
            writer=DummyWriter(),
            checkpointer=None
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self._target = target or TrivialTarget()
        self._scheduler = scheduler
        self._target.init(model)
        self._updates = 0
        self._optimizer = optimizer
        self._loss = loss
        self._loss_scaling = loss_scaling
        self._cache = []
        self._clip_grad = clip_grad
        self._writer = writer
        self._name = name

        if checkpointer is None:
            checkpointer = PeriodicCheckpointer(DEFAULT_CHECKPOINT_FREQUENCY)
        self._checkpointer = checkpointer
        self._checkpointer.init(
            self.model,
            os.path.join(writer.log_dir, name + '.pt')
        )

    def __call__(self, *inputs, detach=True):
        '''
        Run a forward pass of the model.

        If detach=True, the computation graph is cached and the result is detached.
        If detach=False, nothing is cached and instead returns the attached result.
        '''
        result = self.model(*inputs)
        if detach:
            self._enqueue(result)
            return result.detach()
        return result

    def eval(self, *inputs):
        '''Run a forward pass of the model in no_grad mode.'''
        with torch.no_grad():
            return self.model(*inputs)

    def target(self, *inputs):
        '''Run a forward pass of the target network.'''
        return self._target(*inputs)

    def reinforce(self, errors, retain_graph=False):
        '''Update the model using the cache and the errors passed in.'''
        batch_size = len(errors)
        cache = self._dequeue(batch_size)
        if cache.requires_grad:
            loss = self._loss(cache, errors) * self._loss_scaling
            self._writer.add_loss(self._name, loss)
            loss.backward(retain_graph=retain_graph)
            self.step()

    def step(self):
        '''Given that a bakcward pass has been made, run an optimization step.'''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._target.update()
        if self._scheduler:
            self._writer.add_schedule(self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._scheduler.step()
        self._checkpointer()

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

class ConstantLR():
    '''Dummy LRScheduler'''
    def step(self):
        pass
