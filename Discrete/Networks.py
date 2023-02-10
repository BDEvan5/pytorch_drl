
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#Hyper params:
hidden_size = 400


class Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        probs = F.softmax(x, dim=softmax_dim)
        
        return probs
        # dist  = Categorical(prob)
        
        # return dist
    
    
class Critic(nn.Module):
    def __init__(self, obs_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc_v  = nn.Linear(hidden_size, 1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    