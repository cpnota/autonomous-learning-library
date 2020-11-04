import torch
from all.nn import RLNetwork
from .approximation import Approximation


class QContinuous(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='q',
            **kwargs
    ):
        model = QContinuousModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class QContinuousModule(RLNetwork):
    def forward(self, states, actions):
        x = torch.cat((states.observation.float(), actions), dim=1)
        return self.model(x).squeeze(-1) * states.mask.float()
