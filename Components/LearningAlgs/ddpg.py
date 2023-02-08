from Components.Networks import PolicyNet, CriticNet
from Components.ReplayBuffers import ReplayBuffer
from Components.Noises import OrnsteinUhlenbeckNoise
from Components.utils import soft_update

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
tau          = 0.005 # for target network soft update

  
class DDPG:
    def __init__(self, state_dim, action_dim, action_scale):
        self.action_scale = action_scale
        
        self.memory = ReplayBuffer()

        self.q, self.q_target = CriticNet(state_dim, action_dim), CriticNet(state_dim, action_dim)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = PolicyNet(state_dim, action_dim), PolicyNet(state_dim, action_dim)
        self.mu_target.load_state_dict(self.mu.state_dict())


        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def act(self, state):
        action = self.mu(torch.from_numpy(state).float()) * self.action_scale
        action = action.item() + self.ou_noise()[0]
        
        return action
        
      
    def train(self):
        if self.memory.size() < 1000:
            return
        
        for i in range(1):
            s,a,r,s_prime,done_mask  = self.memory.sample(batch_size)
            
            target = r + gamma * self.q_target(s_prime, self.mu_target(s_prime)) * done_mask
            q_loss = F.smooth_l1_loss(self.q(s,a), target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            mu_loss = -self.q(s,self.mu(s)).mean() 
            self.mu_optimizer.zero_grad()
            mu_loss.backward()
            self.mu_optimizer.step()
        
            soft_update(self.q, self.q_target, tau)
            soft_update(self.mu, self.mu_target, tau)
     
        