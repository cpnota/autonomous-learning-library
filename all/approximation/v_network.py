from all.nn import VModule
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
