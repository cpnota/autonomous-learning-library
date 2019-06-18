import torch
from torch.nn.functional import mse_loss
from all.layers import ListNetwork
from .approximation import Approximation

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())
    return _loss

class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            num_actions,
            loss=mse_loss,
            name='q',
            **kwargs
    ):
        model = ListNetwork(model, (num_actions,))
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )

    def __call__(self, states, actions=None):
        result = self._eval(states, actions, self.model)
        self._enqueue(result)
        return result.detach()

    def eval(self, states, actions=None):
        with torch.no_grad():
            training = self.target_model.training
            result = self._eval(states, actions, self.target_model.eval())
            self.target_model.train(training)
            return result

    def _eval(self, states, actions, model):
        values = model(states)
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values.gather(1, actions.view(-1, 1)).squeeze(1)
