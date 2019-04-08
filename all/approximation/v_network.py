import torch
from torch import optim
from torch.nn import utils
from torch.nn.functional import mse_loss
from all.layers import ListNetwork
from .v_function import ValueFunction

class ValueNetwork(ValueFunction):
    def __init__(self, model, optimizer=None, loss=mse_loss, clip_grad=0):
        self.model = ListNetwork(model, (1,))
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.loss = loss
        self.cache = None
        self.clip_grad = clip_grad

    def __call__(self, states):
        result = self.model(states).squeeze(1)
        self.cache = result
        return result.detach()

    def eval(self, states):
        with torch.no_grad():
            training = self.model.training
            result = self.model(states).squeeze(1)
            self.model.train(training)
            return result

    def reinforce(self, td_errors, retain_graph=False):
        if self.cache.requires_grad:
            targets = td_errors + self.cache.detach()
            loss = self.loss(self.cache, targets)
            loss.backward(retain_graph=retain_graph)
            if self.clip_grad != 0:
                utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
