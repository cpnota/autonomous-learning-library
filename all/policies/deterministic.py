from all.approximation import Approximation
from all.nn import ListNetwork


class DeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='policy',
            **kwargs
    ):
        model = ListNetwork(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )
