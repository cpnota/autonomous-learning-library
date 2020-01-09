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
        model = QModuleContinuous(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class QModuleContinuous(RLNetwork):
    def forward(self, states, actions):
        x = torch.cat((states.features.float(), actions), dim=1)
        return super().forward(x).squeeze(-1) * states.mask.float()
