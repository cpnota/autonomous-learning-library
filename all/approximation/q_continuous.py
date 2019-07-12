from torch.nn.functional import mse_loss
from all.nn import QModuleContinuous, td_loss
from .approximation import Approximation

class QContinuous(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            loss=mse_loss,
            name='q',
            **kwargs
    ):
        model = QModuleContinuous(model)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
