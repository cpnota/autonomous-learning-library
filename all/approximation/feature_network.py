import torch
from torch.nn import utils
from all.layers import ListToList
from .features import Features

class FeatureNetwork(Features):
    def __init__(self, model, optimizer, clip_grad=0):
        self.model = ListToList(model)
        self.optimizer = optimizer
        self.clip_grad = clip_grad

    def __call__(self, states):
        return self.model(states)

    def eval(self, states):
        with torch.no_grad():
            training = self.model.training
            result = self.model(states)
            self.model.train(training)
            return result

    def reinforce(self):
        # loss comes from elsewhere
        if self.clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
