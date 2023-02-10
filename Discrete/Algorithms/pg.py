import torch
import torch.optim as optim

from Discrete.Networks import Actor, Critic
from Discrete.OnPolicyBuffer import OnPolicyBuffer
    
gamma = 0.99
    
class PolicyGradient:
    def __init__(self, num_inputs, num_outputs) -> None:
        self.actor = Actor(num_inputs, num_outputs)
        self.optimizer = optim.Adam(list(self.actor.parameters()), lr=lr)
        self.buffer = OnPolicyBuffer()
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy(), log_prob
        
    def compute_rewards_to_go(self):
        R = 0 # This is a problem for a fixed rollout size...
        #! eish
        returns = []
        for step in reversed(range(len(self.buffer.rewards))):
            R = torch.FloatTensor(self.buffer.rewards[step]) + gamma * R * torch.FloatTensor(self.buffer.done_masks[step])
            returns.insert(0, R)
            
        return returns
        
    def train(self):
        returns = self.compute_rewards_to_go()

        states = torch.FloatTensor(self.buffer.states)
        actions = torch.IntTensor(self.buffer.actions)
        
        probs = self.actor.pi(states, softmax_dim=1)
        probs = probs.gather(1, actions.long())
        log_probs = torch.log(probs)[:, 0]
        
        log_probs = torch.stack(self.buffer.log_probs)
        returns   = torch.cat(returns).detach()
        
        actor_loss  = -(log_probs * returns.detach()).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        self.buffer.reset()
    