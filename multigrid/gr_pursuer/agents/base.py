class BaseAgent():

    def __init__(self, agent):
        self.agent = agent

    def compute_action(self, obs):
        raise NotImplementedError
    