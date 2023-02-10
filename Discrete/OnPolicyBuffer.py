import numpy as np


# class OnPolicyBuffer:
#     def __init__(self):
#         self.log_probs = []
#         self.values    = []
#         self.rewards   = []
#         self.masks     = []
        
#     def add(self, log_prob, values, reward, mask):
#         self.log_probs.append(log_prob)
#         self.values.append(values)
#         self.rewards.append(reward)
#         self.masks.append(mask)
        
#     def compute_rewards_to_go(self, next_value, gamma=0.99):
#         R = next_value
#         returns = []
#         for step in reversed(range(len(self.rewards))):
#             R = self.rewards[step] + gamma * R * self.masks[step]
#             returns.insert(0, R)
            
#         return returns
    
#     def reset(self):
#         self.log_probs = []
#         self.values    = []
#         self.rewards   = []
#         self.masks     = []
   
   
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
   
   

   