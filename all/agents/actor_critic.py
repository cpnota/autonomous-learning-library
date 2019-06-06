from .abstract import Agent

class ActorCritic(Agent):
    def __init__(self, v, policy, gamma=1):
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.previous_state = None

    def act(self, state, reward):
        if self.previous_state:
            td_error = (
                reward
                + self.gamma * self.v.eval(state)
                - self.v(self.previous_state)
            )
            self.v.reinforce(td_error)
            self.policy.reinforce(td_error)
        self.previous_state = state
        return self.policy(state)
