from .abstract import Agent

class ActorCritic(Agent):
    def __init__(self, v, policy, gamma=1):
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.previous_state = None

    def initial(self, state, info=None):
        self.previous_state = state
        return self.policy(state)

    def act(self, state, reward, info=None):
        td_error = reward + self.gamma * self.v.eval(state) - self.v(self.previous_state)
        self.v.reinforce(td_error)
        self.policy.reinforce(td_error)
        self.previous_state = state
        return self.policy(state)

    def terminal(self, reward, info=None):
        td_error = reward - self.v(self.previous_state)
        self.v.reinforce(td_error)
        self.policy.reinforce(td_error)
