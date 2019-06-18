import torch
from torch.nn.functional import mse_loss
from all.layers import ListNetwork
from .approximation import Approximation

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())
    return _loss

class ValueNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            loss=mse_loss,
            name='v',
            **kwargs
    ):
        model = ListNetwork(model, (1,))
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )

    def __call__(self, states):
        result = self.model(states).squeeze(1)
        self._enqueue(result)
        return result.detach()

    def eval(self, states):
        with torch.no_grad():
            training = self.target_model.training
            result = self.target_model(states).squeeze(1)
            self.target_model.train(training)
            return result
