import gym


class GymWrapper:
    def __init__(self, environment_name):
        self.env = gym.make(environment_name)
        self.state = None
        self.action = None
        self.reward = None
        self.done = None

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.reward = 0

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done

    def close(self):
        self.env.close()
