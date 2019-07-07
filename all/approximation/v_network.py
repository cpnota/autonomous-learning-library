from torch.nn.functional import mse_loss
from all.layers import VModule
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
        model = VModule(model)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
