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

    def new_episode(self, env):
        self.env = env
        self.state = self.env.state
        self.action = self.policy(self.state)

    def act(self):
        self.env.step(self.action)
        self.next_state = self.env.state
        self.next_action = (
            None if self.env.done
            else self.policy(self.next_state)
        )
        self.update()
        self.state = self.next_state
        self.action = self.next_action

    def update(self):
        td_error = (
            self.env.reward
            + self.gamma * self.q.eval(self.next_state, self.next_action)
            - self.q(self.state, self.action)
        )
        self.q.reinforce(td_error)
