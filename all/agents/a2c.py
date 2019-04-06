from .abstract import Agent

class A2C(Agent):
    def __init__(self, v, policy, steps, gamma=1):
        self.v = v
        self.policy = policy
        self.steps = steps
        self.gamma = gamma
        self.previous_states = None

    def act(self, states, rewards, info=None):
        if self.previous_states is not None:
            td_errors = rewards + self.gamma * self.v.eval(states) - self.v(self.previous_states)
            self.v.reinforce(td_errors)
            self.policy.reinforce(td_errors)
        self.previous_states = states
        return self.policy(states)
