from .abstract import Agent

class VAC(Agent):
    '''Vanilla Actor-Critic'''
    def __init__(self, features, v, policy, gamma=1):
        self.features = features
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self._previous_features = None

    def act(self, state, reward):
        if self._previous_features:
            td_error = (
                reward
                + self.gamma * self.v.eval(self.features.eval(state))
                - self.v(self._previous_features)
            )
            self.v.reinforce(td_error)
            self.policy.reinforce(td_error)
            self.features.reinforce()
        self._previous_features = self.features(state)
        return self.policy(self._previous_features)
