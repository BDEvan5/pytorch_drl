import numpy as np
import torch
import torch.nn.functional as F

from Continuous.Networks import PolicyNet, CriticNet
from Continuous.ReplayBuffers import ReplayBuffer, SmartBuffer

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class TD3(object):
    def __init__(self, state_dim, action_dim, action_scale):
        self.action_scale = action_scale
        self.act_dim = action_dim
        
        self.actor = PolicyNet(state_dim, action_dim, action_scale)
        self.actor_target = PolicyNet(state_dim, action_dim, action_scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = CriticNet(state_dim, action_dim)
        self.critic_target_1 = CriticNet(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = CriticNet(state_dim, action_dim)
        self.critic_target_2 = CriticNet(state_dim, action_dim)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.memory = SmartBuffer(state_dim, action_dim)

    def act(self, state, noise=EXPLORE_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.action_scale, self.action_scale)

    def train(self, iterations=2):
        if self.memory.size() < BATCH_SIZE:
            return
        for it in range(iterations):
            state, action, next_state, reward, done = self.memory.sample(BATCH_SIZE)
            self.update_critic(state, action, next_state, reward, done)
        
            if it % POLICY_FREQUENCY == 0:
                self.update_policy(state)
                
                soft_update(self.critic_1, self.critic_target_1, tau)
                soft_update(self.critic_2, self.critic_target_2, tau)
                soft_update(self.actor, self.actor_target, tau)
    
    def update_critic(self, state, action, next_state, reward, done):
        noise = torch.normal(torch.zeros(action.size()), POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.action_scale, self.action_scale)

        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_policy(self, state):
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                


        
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)