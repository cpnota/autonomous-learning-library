from all.nn import QModuleContinuous
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
