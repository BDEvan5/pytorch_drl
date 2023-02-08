
import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_fc1_units = 400
hidden_fc2_units = 300

class PolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(PolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_fc1_units)
        self.fc2 = nn.Linear(hidden_fc1_units, hidden_fc2_units)
        self.fc_mu = nn.Linear(hidden_fc2_units, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu

class CriticNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(CriticNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, hidden_fc1_units)
        self.fc2 = nn.Linear(hidden_fc1_units, hidden_fc2_units)
        self.fc_out = nn.Linear(hidden_fc2_units, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q


if __name__ == "__main__":
    print("Hello World!")
    
    