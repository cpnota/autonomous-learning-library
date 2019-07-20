# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from all.agents import VAC
from all.approximation import VNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def vac(
        discount_factor=0.99,
        lr_v=5e-3,
        lr_pi=1e-3,
        alpha=0.99, # RMSprop momentum decay
        eps=1e-5,   # RMSprop stability
        device=torch.device('cpu')
):
    def _vac(env, writer=DummyWriter()):
        value_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)
        feature_model = fc_relu_features(env).to(device)

        value_optimizer = RMSprop(value_model.parameters(), lr=lr_v, alpha=alpha, eps=eps)
        policy_optimizer = RMSprop(policy_model.parameters(), lr=lr_pi, alpha=alpha, eps=eps)
        feature_optimizer = RMSprop(feature_model.parameters(), lr=lr_pi, alpha=alpha, eps=eps)

        v = VNetwork(value_model, value_optimizer, writer=writer)
        policy = SoftmaxPolicy(policy_model, policy_optimizer, env.action_space.n, writer=writer)
        features = FeatureNetwork(feature_model, feature_optimizer)

        return VAC(features, v, policy, gamma=discount_factor)
    return _vac
