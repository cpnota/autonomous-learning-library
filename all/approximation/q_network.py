from torch.nn.functional import mse_loss
from all.nn import QModule
from .approximation import Approximation

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
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
