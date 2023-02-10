from Continuous.Networks import PolicyNet, CriticNet
from Continuous.ReplayBuffers import ReplayBuffer, SmartBuffer
from Continuous.Noises import OrnsteinUhlenbeckNoise
from Continuous.utils import soft_update

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
        
        self.memory = SmartBuffer(state_dim, action_dim)

        self.critic = CriticNet(state_dim, action_dim)
        self.critic_target = CriticNet(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = PolicyNet(state_dim, action_dim, action_scale)
        self.actor_target = PolicyNet(state_dim, action_dim, action_scale)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.mu_optimizer = optim.Adam(self.actor.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.critic.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def act(self, state):
        action = self.actor(torch.from_numpy(state).float()) * self.action_scale
        action = action.detach().numpy() + self.ou_noise()
        
        return action
        
      
    def train(self):
        if self.memory.size() < 1000:
            return
        
        for i in range(1):
            s, a, s_prime, r, done_mask  = self.memory.sample(batch_size)
            
            target = r + gamma * self.critic_target(s_prime, self.actor_target(s_prime)) * done_mask
            q_loss = F.smooth_l1_loss(self.critic(s,a), target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            mu_loss = -self.critic(s,self.actor(s)).mean() 
            self.mu_optimizer.zero_grad()
            mu_loss.backward()
            self.mu_optimizer.step()
        
            soft_update(self.critic, self.critic_target, tau)
            soft_update(self.actor, self.actor_target, tau)
     
        