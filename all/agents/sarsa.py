from all.agents.agent import Agent

class Sarsa(Agent):
    def __init__(self, action_approximation, policy):
        self.action_approximation = action_approximation
        self.policy = policy

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
        self.next_action = self.policy(self.next_state)
        self.update()
        self.state = self.next_state
        self.action = self.next_action

    def update(self):
        td_error = (self.env.reward
                    + (self.action_approximation(self.next_state, self.next_action)
                       if not self.env.done else 0)
                    - self.action_approximation(self.state, self.action))

        self.action_approximation.update(td_error, self.state, self.action)
