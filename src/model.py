import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
  def __init__(self, obs_dim, act_dim, hidden_state=(64, 32)):
    super().__init__()

    self.shared = nn.Sequential(
        nn.Linear(obs_dim, hidden_state[0]),
        nn.ReLU(),
        nn.Linear(hidden_state[0], hidden_state[1]),
        nn.ReLU()
    )

    self.mean_layer = nn.Linear(hidden_state[1], act_dim)
    self.log_std = nn.Parameter(torch.zeros(act_dim))

    self.value = nn.Linear(hidden_state[1], 1)

  def forward(self, obs):
    features = self.shared(obs)

    mean = self.mean_layer(features)
    std = torch.exp(self.log_std)
    dist = Normal(mean, std)

    value = self.value(features)

    return dist, value

  def get_action(self, obs):
    dist, value = self.forward(obs)

    action = dist.sample()
    log_prob = dist.log_prob(action).sum(axis=-1)
    return action, log_prob, value

  def evaluate_actions(self, obs, action):
    dist, value = self.forward(obs)

    log_prob = dist.log_prob(action).sum(axis=-1)
    entropy = dist.entropy().sum(axis=-1)
    return log_prob, entropy, value