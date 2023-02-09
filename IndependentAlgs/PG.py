import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#Hyper params:
# hidden_size = 256
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
   
class BufferPG:
    def __init__(self):
        self.log_probs = []
        self.rewards   = []
        self.masks     = []
        
    def add(self, log_prob, reward, mask):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.masks.append(mask)
        
    def compute_rewards_to_go(self, gamma=0.99):
        R = torch.FloatTensor([0.0]) # it should start at the end of an ep with a terminal reward
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
            
        return returns
    
    def reset(self):
        self.log_probs = []
        self.rewards   = []
        self.masks     = []
   
    
class PolicyGradient:
    def __init__(self, num_inputs, num_outputs) -> None:
        self.actor = Actor(num_inputs, num_outputs)
        self.optimizer = optim.Adam(list(self.actor.parameters()), lr=lr)
        self.buffer = BufferPG()
        
    def act(self, state):
        state = torch.FloatTensor(state)
        dist = self.actor.pi(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy(), log_prob
        
    def train(self):
        returns = self.buffer.compute_rewards_to_go()
        log_probs = torch.stack(self.buffer.log_probs)
        returns   = torch.cat(returns).detach()
        
        actor_loss  = -(log_probs * returns.detach()).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        self.buffer.reset()
    
def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001)


def test_policy_gradient():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    agent = PolicyGradient(num_inputs=num_inputs, num_outputs=num_outputs)
    
    max_frames   = 50000
    frame_idx    = 0
    state = env.reset()
    training_rewards = []
    ep_reward = 0

    while frame_idx < max_frames:
        i = 0
        while True:
            action, log_prob = agent.act(state)

            next_state, reward, done, _ = env.step(action)
    
            agent.buffer.add(log_prob, reward, 1 - done)
            ep_reward += reward
        
            state = next_state
            frame_idx += 1
        
            if done:
                print(f"{frame_idx} -> Episode reward: ", ep_reward)
                training_rewards.append(ep_reward)
                ep_reward = 0
                state = env.reset()
                if i > num_steps:
                    break
                
            if frame_idx % 1000 == 0:
                plot(frame_idx, training_rewards)
            
            i += 1
            
        agent.train()
    
test_policy_gradient()