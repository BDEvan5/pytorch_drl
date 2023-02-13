import gym
import random
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
ENV_NAME = "CartPole-v1"

GAMMA = 0.94
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class SmartBufferDQN(object):
    def __init__(self, state_dim=4, max_size=MEMORY_SIZE):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.ptr = 0

        self.states = np.empty((max_size, state_dim))
        self.actions = np.empty((max_size, 1))
        self.next_states = np.empty((max_size, state_dim))
        self.rewards = np.empty((max_size, 1))
        self.dones = np.empty((max_size, 1))

    def add(self, s, a, s_p, r, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = s_p
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d

        self.ptr += 1
        
        if self.ptr == 99999: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, 1))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr


class Qnet(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
      
class DQN:
    def __init__(self, obs_space, action_space, name="Agent"):
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory = SmartBufferDQN(obs_space)

        self.name = name
        self.model = None 
        self.target = None
        self.optimizer = None

        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

    def create_agent(self, h_size):
        obs_space = self.obs_space
        action_space = self.action_space

        self.model = Qnet(obs_space, action_space, h_size)
        self.target = Qnet(obs_space, action_space, h_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def sample_action(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
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
            s, a, s_p, r, done = self.memory.sample(BATCH_SIZE)

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



def test_cartpole():
    env = gym.make('CartPole-v1')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n, "AgentCartpole")
    dqn.create_agent(100)

    print_n = 20
    rewards = []
    for n in range(200):
        score, done, state = 0, False, env.reset()
        while not done:
            a = dqn.sample_action(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            dqn.memory.add(state, a, s_prime, r/100, done_mask)
            state = s_prime
            score += r
            dqn.experience_replay()

        rewards.append(score)
        if n % print_n == 1:
            print(f"Run: {n} --> Score: {score} --> Mean: {np.mean(rewards[-20:])} --> exp: {dqn.exploration_rate}")

if __name__ == '__main__':
    test_cartpole()

