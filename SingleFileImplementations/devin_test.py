import math
import random
import sys
​
import gym
import numpy as np
import time 
​
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from matplotlib import pyplot as plt
​
torch.autograd.set_detect_anomaly(True)
​
# Hyper params:


MEMORY_SIZE = 100000
SEED = 0
TAU = 1e-2
GAMMA = 0.99
BATCH_SIZE = 100
LR = 1e-3
​
​
NN_LAYER_1 = 400
NN_LAYER_2 = 300
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6
​
class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetworkSAC, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
            
        return action, log_prob
   
​
class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)
​
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q
​
​
class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
​
        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))
​
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
​
        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0
​
    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))
​
        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)
​
        return states, actions, next_states, rewards, dones
​
    def size(self):
        return self.ptr
   
class SAC(object):
    def __init__(self, state_dim, action_dim):
        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)
​
        self.soft_q_net1 = DoubleQNet(state_dim, action_dim)
        self.soft_q_net2 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net1 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net2 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net1.load_state_dict(self.soft_q_net1.state_dict())
        self.target_soft_q_net2.load_state_dict(self.soft_q_net2.state_dict())
        #torch.loss()
        
        self.policy_net = PolicyNetworkSAC(state_dim, action_dim)
        
        self.soft_q_criterion = nn.MSELoss()
        self.q_optimiser = optim.Adam(list(self.soft_q_net1.parameters()) + list(self.soft_q_net2.parameters()), lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.target_entropy = -np.prod(action_dim).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ =  self.policy_net(state)
        return action.detach()[0].numpy()
               
    def train(self, iterations=2):
        for _ in range(0, iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            alpha = self.log_alpha.exp()
            
            self.update_policy(state, alpha)
            self.update_Q(state, action, next_state, reward, done, alpha)
            self.update_alpha(state)
            
            soft_update(self.soft_q_net1, self.target_soft_q_net1, TAU)
            soft_update(self.soft_q_net2, self.target_soft_q_net2, TAU)
        
    def update_policy(self, state, alpha):
        new_actions, log_pi = self.policy_net(state)
​
        q1 = self.soft_q_net1(state, new_actions)
        q2 = self.soft_q_net2(state, new_actions)
        q_new_actions = torch.min(q1, q2)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def update_Q(self, state, action, next_state, reward, done, alpha):
            current_q1 = self.soft_q_net1(state, action)
            current_q2 = self.soft_q_net2(state, action)
​
            new_next_actions, new_log_pi= self.policy_net(next_state)
​
            target_q1 = self.target_soft_q_net1(next_state, new_next_actions)
            target_q2 = self.target_soft_q_net2(next_state, new_next_actions)
            target_q_values = torch.min(target_q1, target_q2) - alpha * new_log_pi
​
            q_target = reward + done * GAMMA * target_q_values
            q_loss = self.soft_q_criterion(current_q1, q_target.detach()) + self.soft_q_criterion(current_q2, q_target.detach())
            
            self.q_optimiser.zero_grad()
            q_loss.backward()
            self.q_optimiser.step()
           
    def update_alpha(self, state):
        new_actions, log_pi = self.policy_net(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
                
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
   
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
​
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action
​
def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001) 
​
def observe(env, replay_buffer, observation_steps):
    time_steps = 0
    (state,_) = env.reset()
    done = False
​
    while time_steps < observation_steps:
        action = env.action_space.sample()
        next_state, reward, terminated,truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, next_state, reward, done)  
​
        state = next_state
        time_steps += 1
​
        if done:
            (state,_) = env.reset()
            done = False
​
        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()
​
    print("")
​
def OffPolicyTrainingLoop(agent, env, training_steps=10000, view=True):
    lengths, rewards = [], []
    (state,_), done = env.reset(), False
    ep_score, ep_steps = 0, 0
    for t in range(1, training_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        done = 0 if ep_steps + 1 == 200 else float(done)
        agent.replay_buffer.add(state, action, next_state, reward, done)  
        ep_score += reward
        ep_steps += 1
        state = next_state
        
        agent.train()
        
        if done:
            lengths.append(ep_steps)
            rewards.append(ep_score)
            (state,_), done = env.reset(), False
            print("Step: {}, Episode :{}, Score : {:.1f}".format(t, len(lengths), ep_score))
            ep_score, ep_steps = 0, 0
        
        
        if t % 1000 == 0 and view:
            plot(t, rewards)
        
    return lengths, rewards
​
def test_sac():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env = NormalizedActions(env)
    
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    
    observe(env, agent.replay_buffer, 10000)
    OffPolicyTrainingLoop(agent, env, 10000)
    
if __name__ == '__main__':
    test_sac()