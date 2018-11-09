import numpy as np

# TODO implement function approximator interface
class Greedy:
    def __init__(self, action_approximation, epsilon=0.1):
        self.action_approximation = action_approximation
        self.epsilon = epsilon

    def choose_action(self, state):
        action_scores = self.action_approximation.call(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_scores.shape[0])
        return np.argmax(action_scores)
