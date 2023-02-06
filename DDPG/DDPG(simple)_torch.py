import numpy as np
from collections import deque
import random
import gym

import torch
import torch.nn as nn 
import torch.nn.functional as F 

class BasicBuffer_a:
    def __init__(self, size, obs_dim, act_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = np.asarray([rew])
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict= dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])
        return (temp_dict['s'],temp_dict['a'],temp_dict['r'].reshape(-1,1),temp_dict['s2'],temp_dict['d'])

class Critic_gen(nn.Module):
    def __init__(self):
        super(Critic_gen, self).__init__()

        self.fc1 = nn.Linear(4, 1204)
        self.fc2 = nn.Linear(1204, 512)
        self.fc3 = nn.Linear(512, 300)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        obs = torch.cat(x, axis=-1)

        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        v =  self.fc4(h3)

        return v

class Actor_gen(nn.Module):
    def __init__(self, max_act):
        super(Actor_gen, self).__init__()

        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 128)
        self.fc4 = nn.Linear(128, 1)

        self.max_act = max_act

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        a =  torch.tanh(self.fc4(h3))

        act = torch.mul(a, self.max_act)

        return act


class DDPGAgent:
    def __init__(self):
        self.obs_dim = 3
        self.action_dim = 1 # x and y for target
        self.action_max = 2 # will be scaled later
        
        # hyperparameters
        self.gamma = 0.99
        self.tau = 1e-2

        # Main network outputs
        self.mu = Actor_gen(2)
        self.q_mu = Critic_gen()

        # Target networks
        self.mu_target = Actor_gen(2)
        self.q_mu_target = Critic_gen()
      
        # Copying weights in,
        self.mu.load_state_dict(self.mu_target.state_dict())
        self.q_mu_target.load_state_dict(self.q_mu.state_dict())
    
        # optimizers
        self.mu_optimizer = torch.optim.Adam(self.mu.parameters(), lr=1e-3)
        self.q_mu_optimizer = torch.optim.Adam(self.q_mu.parameters(), lr=1e-3)

        self.replay_buffer = BasicBuffer_a(100000, obs_dim=3, act_dim=1)
        
        self.q_losses = []
        self.mu_losses = []
        
    def get_action(self, s, noise_scale):
        s = torch.tensor(s, dtype=torch.float)
        a =  self.mu(s).detach().numpy()
        a += noise_scale * np.random.randn(self.action_dim)
        act = np.clip(a, -self.action_max, self.action_max)

        return act

    def train(self):
        batch_size = 32
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)

        X = torch.tensor(X, dtype=torch.float)
        A = torch.tensor(A, dtype=torch.float)
        X2 = torch.tensor(X2, dtype=torch.float)
        R = torch.tensor(R, dtype=torch.float)

        # Updating Ze Critic
        A2 =  self.mu_target(X2)
        q_target = R + self.gamma * self.q_mu_target([X2,A2]).detach()
        qvals = self.q_mu([X,A]) 
        q_loss = ((qvals - q_target)**2).mean()
        self.q_mu_optimizer.zero_grad()
        q_loss.backward()
        self.q_mu_optimizer.step()
        self.q_losses.append(q_loss)

        #Updating ZE Actor
        A_mu =  self.mu(X)
        Q_mu = self.q_mu([X,A_mu])
        mu_loss =  - Q_mu.mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        soft_update(self.mu, self.mu_target, self.tau)
        soft_update(self.q_mu, self.q_mu_target, self.tau)


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def test(env, agent):
    episode_rewards = []

    noise = 0.1
    for episode in range(20):
        state = env.reset()
        episode_reward = 0

        for step in range(500):
            action = agent.get_action(state, noise)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == 499 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > 32:
                agent.train()   

            if done or step == 499:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    
    agent = DDPGAgent()
    episode_rewards = test(env, agent)



