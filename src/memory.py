class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.log_probs = []
    self.rewards = []
    self.dones = []
    self.values = []

  def store(self, state, action, log_prob, reward, done, value):
    self.states.append(state)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.rewards.append(reward)
    self.dones.append(done)
    self.values.append(value)

  def clear(self):
    self.__init__()

def calculate_gae(rewards, values, dones, gamma=0.99, lamb=0.95):
  advantage = []
  gae = 0
  values = values + [0]

  for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
    gae = delta + gamma * lamb * (1-dones[t]) * gae
    advantage.insert(0, gae)

  returns = [adv + v for adv, v in zip(advantage, values[:-1])]

  return advantage, returns