import os
from datetime import datetime
import torch
from torch import optim
from torch.nn import utils
from torch.nn.functional import mse_loss
from tensorboardX import SummaryWriter
from all.layers import ListNetwork
from .v_function import ValueFunction

class ValueNetwork(ValueFunction):
    def __init__(self, model, optimizer=None, loss=mse_loss, clip_grad=0):
        self.model = ListNetwork(model, (1,))
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.loss = loss
        self._cache = []
        self.clip_grad = clip_grad
        log_dir = os.path.join(
            'runs', str(datetime.now())
        )
        self._writer = SummaryWriter(log_dir=log_dir)
        self._count = 0

    def __call__(self, states):
        result = self.model(states).squeeze(1)
        self._cache.append(result)
        return result.detach()

    def eval(self, states):
        with torch.no_grad():
            training = self.model.training
            result = self.model(states).squeeze(1)
            self.model.train(training)
            return result

    def reinforce(self, td_errors, retain_graph=False):
        td_errors = td_errors.view(-1)
        batch_size = len(td_errors)
        cache = self.decache(batch_size)

        if cache.requires_grad:
            targets = td_errors + cache.detach()
            loss = self.loss(cache, targets)
            self._writer.add_scalar('value_loss', loss, self._count)
            self._count += 1
            # pylint: disable=len-as-condition
            loss.backward(retain_graph=retain_graph)
            if self.clip_grad != 0:
                utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def decache(self, batch_size):
        i = 0
        items = 0
        while items < batch_size:
            items += len(self._cache[i])
            i += 1
        if items != batch_size:
            raise ValueError("Incompatible batch size.")

        cache = torch.cat(self._cache[:i])
        self._cache = self._cache[i:]

        return cache
