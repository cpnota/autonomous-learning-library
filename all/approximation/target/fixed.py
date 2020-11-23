import copy
import torch
from .abstract import TargetNetwork


class FixedTarget(TargetNetwork):
    def __init__(self, update_frequency):
        self._source = None
        self._target = None
        self._updates = 0
        self._update_frequency = update_frequency

    def __call__(self, *inputs):
        with torch.no_grad():
            return self._target(*inputs)

    def init(self, model):
        self._source = model
        self._target = copy.deepcopy(model)

    def update(self):
        self._updates += 1
        if self._should_update():
            self._target.load_state_dict(self._source.state_dict())

    def _should_update(self):
        return self._updates % self._update_frequency == 0
