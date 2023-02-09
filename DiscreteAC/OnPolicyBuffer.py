


class OnPolicyBuffer:
    def __init__(self):
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        
    def add(self, log_prob, values, reward, mask):
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.rewards.append(reward)
        self.masks.append(mask)
        
    def compute_rewards_to_go(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
            
        return returns
    
    def reset(self):
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
   