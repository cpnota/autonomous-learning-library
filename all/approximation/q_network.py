from torch.nn.functional import mse_loss
from all.layers import QModule
from .approximation import Approximation

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())
    return _loss

class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            num_actions,
            loss=mse_loss,
            name='q',
            **kwargs
    ):
        model = QModule(model, num_actions)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
