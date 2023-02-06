
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = torch.device("cpu")
from common.multiprocessing_env import SubprocVecEnv

# num_envs = 16
env_name = "CartPole-v1"

# def make_env():
#     def _thunk():
#         env = gym.make(env_name)
#         return env

#     return _thunk

# envs = [make_env() for i in range(num_envs)]
# envs = SubprocVecEnv(envs)

env = gym.make(env_name)


class Actor(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc_pi = nn.Linear(h_size, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        dist  = Categorical(prob)
        
        return dist
    
class Critic(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc_v  = nn.Linear(h_size,1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


# class ActorCritic(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size):
#         super(ActorCritic, self).__init__()
        
#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        
#         self.actor = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_outputs),
#             nn.Softmax(dim=1),
#         )
        
#     def forward(self, x):
#         value = self.critic(x)
#         probs = self.actor(x)
#         dist  = Categorical(probs)
#         return dist, value
    
def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001)
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        
    print(f"Total reward: {total_reward}")
    
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n

#Hyper params:
hidden_size = 256
lr          = 3e-4
num_steps   = 5

# model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
actor = Actor(num_inputs, num_outputs, hidden_size).to(device)
critic = Critic(num_inputs, num_outputs, hidden_size).to(device)

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
max_frames   = 40000
frame_idx    = 0
test_rewards = []
state = env.reset()
frame_idx = 0

while frame_idx < max_frames:

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        # dist, value = model(state)
        dist = actor.pi(state)
        value = critic.v(state)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)
            
    next_state = torch.FloatTensor(next_state).to(device)
    # _, next_value = model(next_state)
    next_value = critic.v(next_state)
    returns = compute_returns(next_value, rewards, masks)
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
test_env(True)