import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#Hyper params:
NN_LAYER_1 = 400
lr          = 3e-4
num_steps   = 100
gamma = 0.99


class SingleActor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(SingleActor, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc_pi = nn.Linear(NN_LAYER_1, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        probs = F.softmax(x, dim=softmax_dim)
        
        return probs
    

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



    
class PolicyGradient:
    def __init__(self, state_dim, action_dim, n_steps) -> None:
        self.actor = SingleActor(state_dim, action_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()), lr=lr)
        self.replay_buffer = OnPolicyBuffer(state_dim, 10000)
        
    def compute_rewards_to_go(self, rewards, done_masks):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * done_masks[step]
            returns.insert(0, R)
            
        return returns
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.numpy()
        
    def train(self, next_state=None):
        states, actions, next_states, rewards, done_masks = self.replay_buffer.make_data_batch()
        
        returns = self.compute_rewards_to_go(rewards, done_masks)
        returns   = torch.cat(returns).detach()
        
        probs = self.actor.pi(states, softmax_dim=1)
        probs = probs.gather(1, actions.long())
        log_probs = torch.log(probs)[:, 0]
        
        actor_loss  = -(log_probs * returns.detach()).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        self.replay_buffer.reset()
    

def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001) 
    
def OnPolicyTrainingLoop_eps(agent, env, batch_eps=1, view=False):
    frame_idx    = 0
    training_rewards = []
    cum_lengths = []
    ep_reward = 0

    while frame_idx < 50000:
        for ep in range(batch_eps):
            (state, _), done = env.reset(), False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.add(state, action, next_state, reward/100, done)
        
                ep_reward += reward
                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0 and view:
                    plot(frame_idx, training_rewards)
        
            print(f"{frame_idx} -> Episode reward: ", ep_reward)
            training_rewards.append(ep_reward)
            cum_lengths.append(frame_idx)
            ep_reward = 0
    
        agent.train()

    return cum_lengths, training_rewards

     
def test_pg():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PolicyGradient(state_dim, n_acts, num_steps)
    
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True)

if __name__ == "__main__":
    test_pg()