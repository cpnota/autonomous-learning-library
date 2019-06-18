from torch import nn
from torch.nn.functional import mse_loss
from all.layers import ListNetwork
from .approximation import Approximation

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())
    return _loss

class VModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = ListNetwork(model, (1,))

    def forward(self, states):
        return self.model(states).squeeze(-1)

class ValueNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            loss=mse_loss,
            name='v',
            **kwargs
    ):
        model = VModule(model)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
