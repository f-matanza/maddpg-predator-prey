import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # continuous actions for MPE: forces usually mapped to [0, 1]
        return torch.sigmoid(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        # Input dim will be provided by the controller:
        # - Independent DDPG: obs_dim + act_dim
        # - MADDPG: sum(all_obs_dims) + sum(all_act_dims)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,       128)
        self.fc3 = nn.Linear(128,         1)

    def forward(self, state_action):
        """state_action is a concatenated vec of obs & actions"""
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
