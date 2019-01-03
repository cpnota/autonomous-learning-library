from all.agents.agent import Agent


class ActorCritic(Agent):
    def __init__(self, v, policy):
        self.v = v
        self.policy = policy

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
        td_error = self.env.reward + self.v(self.next_state) - self.v(self.state)
        self.v.update(td_error, self.state)
        self.policy.update(td_error, self.state, self.action)
