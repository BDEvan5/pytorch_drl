import gym
import collections
import random
import sys
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class DQN:
    def __init__(self, obs_space, action_space):
        self.model = Qnet(obs_space, action_space)
        self.target = Qnet(obs_space, action_space)
        self.memory = ReplayBuffer()
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.update_steps = 0

    def sample_action(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0,1)
        else: 
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()


    def experience_replay(self):
        n_train = 1
        for i in range(n_train):
            if self.memory.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = self.memory.sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def observe(env, memory, n_itterations=10000):
    s = env.reset()
    done = False
    for i in range(n_itterations):
        action = env.action_space.sample()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        memory.put((s, action, r/100, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()

def test_cartpole():
    env = gym.make('CartPole-v1')
    dqn = DQN(env.observation_space.shape[0], env.action_space.shape)

    print_n = 20

    rewards = []
    observe(env, dqn.memory)
    for n in range(500):
        score, done, state = 0, False, env.reset()
        while not done:
            a = dqn.sample_action(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            dqn.memory.put((state, a, r/100, s_prime, done_mask))
            state = s_prime
            score += r
            dqn.experience_replay()
            
        rewards.append(score)
        if n % print_n == 1:
            print(f"Run: {n} --> Score: {score} --> Mean: {np.mean(rewards[-20:])} --> exp: {dqn.exploration_rate}")



if __name__ == '__main__':
    test_cartpole()

