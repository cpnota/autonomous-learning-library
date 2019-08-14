import torch
from all.nn import QModule
from .approximation import Approximation

class QDist(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            num_actions,
            num_atoms,
            v_min,
            v_max,
            name='q_dist',
            **kwargs
    ):
        model = QModule(model, num_actions * num_atoms)
        self._init_atoms(num_atoms, v_min, v_max)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def _init_atoms(self, num_atoms, v_min, v_max):
        self.atoms = torch.linspace(v_min, v_max, steps=num_atoms)
