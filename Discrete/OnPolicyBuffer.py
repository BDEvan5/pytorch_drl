import numpy as np
import torch
   
   
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


   