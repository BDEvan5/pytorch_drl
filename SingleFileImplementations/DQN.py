import numpy as np
import random 
import gym 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NN_LAYER_1 = 400
NN_LAYER_2 = 300
MEMORY_SIZE = 100000

GAMMA = 0.94
LEARNING_RATE = 0.001
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class QNetworkDQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(QNetworkDQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc3 = nn.Linear(NN_LAYER_2, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
   
   
class DQN:
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space
        self.replay_buffer = OffPolicyBuffer(obs_space, 1)

        self.model = QNetworkDQN(obs_space, action_space)
        self.target = QNetworkDQN(obs_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            state = torch.from_numpy(state).float()
            q_values = self.model.forward(state)
            action = q_values.argmax().item() 
            return action
        
    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE: return
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        next_values = self.target.forward(next_state)
        max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
        q_target = reward + GAMMA * max_vals * done
        q_vals = self.model.forward(state)
        current_q_a = q_vals.gather(1, action.type(torch.int64))
        
        loss = torch.nn.functional.mse_loss(current_q_a, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 100 == 1: 
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
            
        if self.update_steps % 1000 == 1:
            print("Exploration rate: ", self.exploration_rate)



def OffPolicyTrainingLoop(agent, env, training_steps=10000, view=True):
    
    lengths, rewards = [], []
    state, done = env.reset(), False
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
            state, done = env.reset(), False
            print("Step: {}, Episode :{}, Score : {:.1f}".format(t, len(lengths), ep_score))
            ep_score, ep_steps = 0, 0
        
    return lengths, rewards

def test_dqn():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = DQN(state_dim, n_acts)
    
    OffPolicyTrainingLoop(agent, env, 15000)

if __name__ == "__main__":
    test_dqn()