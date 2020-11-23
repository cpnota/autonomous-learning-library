from all.nn import RLNetwork
from .approximation import Approximation


class VNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='v',
            **kwargs
    ):
        model = VModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class VModule(RLNetwork):
    def forward(self, states):
        return super().forward(states).squeeze(-1)
