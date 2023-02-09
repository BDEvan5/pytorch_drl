import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#Hyper params:
hidden_size = 400
lr          = 3e-4
num_steps   = 100


class Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        dist  = Categorical(prob)
        
        return dist
    
    
class Critic(nn.Module):
    def __init__(self, obs_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc_v  = nn.Linear(hidden_size, 1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
class BufferVPG:
    def __init__(self):
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        
    def add(self, log_prob, values, reward, mask):
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.rewards.append(reward)
        self.masks.append(mask)
        
    def compute_rewards_to_go(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
            
        return returns
    
    def reset(self):
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
   
    
class A2C:
    def __init__(self, num_inputs, num_outputs) -> None:
        self.actor = Actor(num_inputs, num_outputs)
        self.critic = Critic(num_inputs)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.entropy = 0
        self.buffer = BufferVPG()
        
    def act(self, state):
        state = torch.FloatTensor(state)
        
        dist = self.actor.pi(state)
        value = self.critic.v(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean()

        return action.numpy(), log_prob, value
        
    def train(self, next_state):
        next_state = torch.FloatTensor(next_state)
        next_value = self.critic.v(next_state)
        returns = self.buffer.compute_rewards_to_go(next_value)

        log_probs = torch.stack(self.buffer.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.buffer.values)
        
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.entropy = 0
        self.buffer.reset()
    
    
def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001)

def test_a2c():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    agent = A2C(num_inputs=num_inputs, num_outputs=num_outputs)
    
    max_frames   = 50000
    frame_idx    = 0
    state = env.reset()
    training_rewards = []
    ep_reward = 0

    while frame_idx < max_frames:
        for _ in range(num_steps):
            action, log_prob, value = agent.act(state)

            next_state, reward, done, _ = env.step(action)
    
            agent.buffer.add(log_prob, value, reward, 1 - done)
            ep_reward += reward
        
            state = next_state
            frame_idx += 1
        
            if done:
                print(f"{frame_idx} -> Episode reward: ", ep_reward)
                training_rewards.append(ep_reward)
                ep_reward = 0
                state = env.reset()
                
            if frame_idx % 1000 == 0:
                plot(frame_idx, training_rewards)
            
        agent.train(next_state)

    
test_a2c()