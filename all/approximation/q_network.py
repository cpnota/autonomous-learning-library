from all.nn import QModule
from .approximation import Approximation

class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            num_actions,
            name='q',
            **kwargs
    ):
        model = QModule(model, num_actions)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )
