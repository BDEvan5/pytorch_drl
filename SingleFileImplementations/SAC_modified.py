import random
import sys

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

torch.autograd.set_detect_anomaly(True)

MEMORY_SIZE = 1000000
SEED = 0
TAU = 1e-2
GAMMA = 0.99
BATCH_SIZE = 100
LR = 1e-3

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

auto_alpha=True # currently set to auto_alpha with stochastic policy

torch.manual_seed(SEED)
np.random.seed(SEED)
   
NN_INIT_W = 3e-3 # not currently used or needed
NN_LAYER_1 = 400
NN_LAYER_2 = 300
    
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetwork, self).__init__()
        
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
        normal = Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON)
            
        return action, mean, log_std, log_prob, std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _, _, _, _ =  self.forward(state)
        return action[0][0]
    
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.linear3 = nn.Linear(NN_LAYER_2, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    
class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

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

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr

    
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    
class SAC(object):
    def __init__(self, state_dim, action_dim):
        self.memory = OffPolicyBuffer(state_dim, action_dim)

        self.soft_q_net1 = CriticNetwork(state_dim, action_dim)
        self.soft_q_net2 = CriticNetwork(state_dim, action_dim)
        self.target_soft_q_net1 = CriticNetwork(state_dim, action_dim)
        self.target_soft_q_net2 = CriticNetwork(state_dim, action_dim)
        self.target_soft_q_net1.load_state_dict(self.soft_q_net1.state_dict())
        self.target_soft_q_net2.load_state_dict(self.soft_q_net2.state_dict())
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        
        self.soft_q_criterion = nn.MSELoss()
        self.q_optimiser = optim.Adam(list(self.soft_q_net1.parameters()) + list(self.soft_q_net2.parameters()), lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.target_entropy = -np.prod(action_dim).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action  = self.policy_net.get_action(state).detach()
        return action.numpy()
           
    def train_alpha(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()
        return alpha
           
    def train(self, iterations):
        for _ in range(0,iterations):
            state, action, next_state, reward, done = self.memory.sample(BATCH_SIZE)

            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_net(state)

            alpha = self.train_alpha(log_pi)

            # Update Policy 
            q1 = self.soft_q_net1(state, new_actions)
            q2 = self.soft_q_net2(state, new_actions)
            q_new_actions = torch.min(q1, q2)
            
            policy_loss = (alpha*log_pi - q_new_actions).mean()

            # Update Soft Q Function
            current_q1 = self.soft_q_net1(state, action)
            current_q2 = self.soft_q_net2(state, action)

            new_next_actions, _, _, new_log_pi, *_ = self.policy_net(next_state)

            target_q1 = self.target_soft_q_net1(next_state, new_next_actions)
            target_q2 = self.target_soft_q_net2(next_state, new_next_actions)
            target_q_values = torch.min(target_q1, target_q2) - alpha * new_log_pi

            q_target = reward + done * GAMMA * target_q_values
            q_loss = self.soft_q_criterion(current_q1, q_target.detach()) + self.soft_q_criterion(current_q2, q_target.detach())
            
            self.q_optimiser.zero_grad()
            q_loss.backward(retain_graph=True)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            
            self.q_optimiser.step()
            self.policy_optimizer.step()

            soft_update(self.soft_q_net1, self.target_soft_q_net1, TAU)
            soft_update(self.soft_q_net2, self.target_soft_q_net2, TAU)
                
        
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
     
     
def observe(env, replay_buffer, observation_steps):
    time_steps = 0
    state = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()
    print("")


def train(total_steps=10000, max_ep_len=500): 
    env = NormalizedActions(gym.make("Pendulum-v1"))
    env.seed(SEED)
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    

    agent = SAC(state_dim, action_dim)

    total_rewards = []
    avg_reward = None
    
    state, reward, done, ep_reward, ep_len, ep_num = env.reset(), 0, False, 0, 0, 1
    
    observe(env, agent.memory, 10000)
    for t in range(1,total_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        done = False if ep_len == max_ep_len else done

        agent.memory.add(state, action, next_state, reward, done)
        ep_reward += reward
        ep_len += 1
        state = next_state
        
        agent.train(2)
        if done or (ep_len == max_ep_len):
            total_rewards.append(ep_reward)
            avg_reward = np.mean(total_rewards[-100:])
            
            print(f"Steps: {t} Episode: {ep_num} Reward: {ep_reward:.2f} Avg Reward: {avg_reward:.2f}")

            state, reward, done, ep_reward, ep_len = env.reset(), 0, False, 0, 0
            ep_num += 1


            

if __name__ == "__main__":
    train()
    