import torch
from all.nn import RLNetwork
from .approximation import Approximation


class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer=None,
            name='q',
            **kwargs
    ):
        model = QModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class QModule(RLNetwork):
    def forward(self, states, actions=None):
        values = super().forward(states)
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values.gather(1, actions.view(-1, 1)).squeeze(1)
