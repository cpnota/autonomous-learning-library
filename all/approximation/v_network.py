from torch.nn.functional import mse_loss
from all.nn import VModule
from .approximation import Approximation

class VNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            loss=mse_loss,
            name='v',
            **kwargs
    ):
        model = VModule(model)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
