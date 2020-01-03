import os
import torch
from torch.nn import utils
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

    def __call__(self, *inputs):
        '''
        Run a forward pass of the model.
        '''
        return self.model(*inputs)

    def eval(self, *inputs):
        '''Run a forward pass of the model in no_grad mode.'''
        with torch.no_grad():
            return self.model(*inputs)

    def target(self, *inputs):
        '''Run a forward pass of the target network.'''
        return self._target(*inputs)

    def reinforce(self, loss):
        loss = self._loss_scaling * loss
        self._writer.add_loss(self._name, loss.detach())
        loss.backward()
        self.step()
        return self

    def step(self):
        '''Given that a backward pass has been made, run an optimization step.'''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._target.update()
        if self._scheduler:
            self._writer.add_schedule(self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._scheduler.step()
        self._checkpointer()
        return self

    def zero_grad(self):
        self._optimizer.zero_grad()
        return self
