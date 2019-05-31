from .abstract import Agent


class Sarsa(Agent):
    def __init__(self, q, policy, gamma=1):
        self.q = q
        self.policy = policy
        self.gamma = gamma
        self.env = None
        self.state = None
        self.action = None
        self.next_state = None
        self.next_action = None

    def initial(self, state):
        self.state = state
        self.action = self.policy(self.state)
        return self.action

    def act(self, next_state, reward):
        next_action = self.policy(next_state)
        td_error = (
            reward
            + self.gamma * self.q.eval(next_state, next_action)
            - self.q(self.state, self.action)
        )
        self.q.reinforce(td_error)
        self.state = next_state
        self.action = next_action
        return self.action

    def terminal(self, state, reward):
        return self.act(state, reward)
