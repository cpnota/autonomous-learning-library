from .abstract import Agent


class ActorCritic(Agent):
    def __init__(self, v, policy, gamma = 1):
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.env = None
        self.state = None
        self.action = None
        self.next_state = None

    def new_episode(self, env):
        self.env = env
        self.next_state = self.env.state

    def act(self):
        self.state = self.next_state
        self.action = self.policy(self.state)
        self.env.step(self.action)
        self.next_state = self.env.state
        self.update()

    def update(self):
        td_error = self.env.reward + self.gamma * self.v.eval(self.next_state) - self.v(self.state)
        self.v.reinforce(td_error)
        self.policy.reinforce(td_error)
