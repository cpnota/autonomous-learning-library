class Sarsa:
  def __init__(self, action_approximation, policy):
    self.action_approximation = action_approximation
    self.policy = policy
  
  def new_episode(self, env):
    self.env = env
    self.state = self.env.state
    self.action = self.policy.choose_action(self.state)

  def act(self):
    self.env.step(self.action)
    self.next_state = self.env.state
    self.next_action = self.policy.choose_action(self.next_state)
    self.update()
    self.state = self.next_state
    self.action = self.next_action

  def update(self):
    td_error = None
    if (self.env.done):
      td_error = self.env.reward - self.action_approximation.call(self.state, self.action)
    else:
      td_error = (
        self.env.reward 
        + self.action_approximation.call(self.next_state, self.next_action) 
        - self.action_approximation.call(self.state, self.action)
      )
    self.action_approximation.update(self.state, self.action, td_error)
