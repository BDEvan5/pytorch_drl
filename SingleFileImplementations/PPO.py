import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 100
h_size = 300

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, h_size)
        self.fc_pi = nn.Linear(h_size, action_dim)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, h_size)
        self.fc_v  = nn.Linear(h_size,1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class OnPolicyBuffer:
    def __init__(self, state_dim, length):
        self.state_dim = state_dim
        self.length = length
        self.reset()
        
        self.ptr = 0
        
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.done_masks[self.ptr] = 1 - done

        self.ptr += 1
    
    def reset(self):
        self.states = np.zeros((self.length, self.state_dim))
        self.actions = np.zeros((self.length, 1), dtype=np.int64)
        self.next_states = np.zeros((self.length, self.state_dim))
        self.rewards = np.zeros((self.length, 1))
        self.done_masks = np.zeros((self.length, 1))
        
        self.ptr = 0
   
   
    def make_data_batch(self):
        states = torch.FloatTensor(self.states[0:self.ptr])
        actions = torch.LongTensor(self.actions[0:self.ptr])
        next_states = torch.FloatTensor(self.next_states[0:self.ptr])
        rewards = torch.FloatTensor(self.rewards[0:self.ptr])
        dones = torch.FloatTensor(self.done_masks[0:self.ptr])
        
        self.reset()
        
        return states, actions, next_states, rewards, dones


class PPO:
    def __init__(self, state_dim, action_dim, name):
        self.name = name
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.network = None
        self.optimizer = None

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
        self.replay_buffer = OnPolicyBuffer(state_dim, 1000)
        
    def act(self, obs):
        prob = self.actor.pi(torch.from_numpy(obs).float())
        m = Categorical(prob)
        a = m.sample().item()

        return a
    
    def calculate_advantage(self, delta):
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
        return advantage
            
    def train(self):
        if self.replay_buffer.ptr < T_horizon:
            return

        s, a, s_prime, r, done_mask = self.replay_buffer.make_data_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.critic.v(s_prime) * done_mask
            delta = td_target - self.critic.v(s)
            delta = delta.detach().numpy()

            advantage = self.calculate_advantage(delta)

            pi = self.actor.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            prob_a = pi_a.clone().detach()
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def load(self, directory="./saves"):
        filename = self.name

        self.network = torch.load('%s/%s_network.pth' % (directory, filename))

        print(f"Agent Loaded: {filename}")

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.network, '%s/%s_network.pth' % (directory, filename))


def main():
    env = gym.make('CartPole-v1')
    model = PPO(4, 2, "ppo_cartpole")
    score = 0.0
    print_interval = 20
    step = 0

    for n_epi in range(1000):
        state = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                action = model.act(state)
                next_state, reward, done, info = env.step(action)

                model.replay_buffer.add(state, action, next_state, reward/100, done)
                # model.replay_buffer.add(state, action, next_state, reward, done)
                state = next_state

                score += reward
                step += 1
                if done:
                    break

            model.train()

        if n_epi%print_interval==0 and n_epi!=0:
            print(f"Step: {step} -> # of episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()