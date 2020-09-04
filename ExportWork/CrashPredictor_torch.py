"""
This page is for predicting the chances of crashing.
Ultimately I want a binary answer, am I going to crash in the next step?
"""
import numpy as np
from collections import deque
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import LibFunctions as lib 

class Policy(nn.Module):
    def __init__(self, in_features):
        super(Policy, self).__init__()
        self.fc_h1 = nn.Linear(in_features, 64)
        self.fc_h2 = nn.Linear(64, 32)
        self.fc_v = nn.Linear(32, 1)

        self.opt_v = optim.RMSprop(self.parameters(), lr=7e-3)

    def forward(self, inputs):
        h1 = F.relu(self.fc_h1(inputs))
        h2 = F.relu(self.fc_h2(h1))
        v = self.fc_v(h2)

        return v


class CrashModel:
    def __init__(self):
        self.policy = Policy(11)
        self.obs_memory = deque()
        self.reward_mem = deque()
        self.full_memory = deque(maxlen=2000)

        self.plot_values = []

    def expect_crash(self, s, t):
        s = torch.tensor(s, dtype=torch.float)
        t = torch.tensor(t, dtype=torch.float)
        obs = torch.cat((s, t), dim=-1)
        v = self.policy(obs)
        self.obs_memory.append(obs)

        if v.item() < 0:
            return True
        return False

    def add_memory_step(self, r, done):
        gamma = 0.99
        self.reward_mem.append(r)
        if done:
            q_val = 0
            self.reward_mem.reverse()
            self.obs_memory.reverse()
            for s, r in zip(self.obs_memory, self.reward_mem):
                q_val = r + gamma * q_val # only happens from done
                self.full_memory.append((s, q_val))
            self.reward_mem.clear()
            self.obs_memory.clear()

    def train(self):
        batch_sz = 32
        if len(self.full_memory) < batch_sz:
            return
        batch = random.sample(self.full_memory, batch_sz)
        states = [b[0] for b in batch]
        states = torch.cat(states, axis=0)
        states = torch.reshape(states, (batch_sz, 11))
        q_vals = [b[1] for b in batch]
        q_vals = torch.tensor(q_vals)

        self.policy.opt_v.zero_grad()
        values = self.policy(states)
        values = torch.squeeze(values, dim=-1)
        value_loss = F.mse_loss(values, q_vals)
        # self.plot_values.append(value_loss.detach().numpy())
        # lib.plot(self.plot_values)
        value_loss.backward()
        self.policy.opt_v.step()

    def run_training(self):
        for i in range(10):
            self.train()


