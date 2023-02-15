import torch
import torch.optim as optim
from Components.Networks import SingleActor
from Components.ReplayBuffers import OnPolicyBuffer


lr          = 3e-4
gamma = 0.99

class PolicyGradient:
    def __init__(self, state_dim, action_dim, n_steps) -> None:
        self.actor = SingleActor(state_dim, action_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()), lr=lr)
        self.buffer = OnPolicyBuffer(state_dim, n_steps)
        
    def compute_rewards_to_go(self, next_value=None):
        # R = next_value
        R = 0
        returns = []
        for step in reversed(range(len(self.buffer.rewards))):
            R = torch.FloatTensor(self.buffer.rewards[step]) + gamma * R * torch.FloatTensor(self.buffer.done_masks[step])
            returns.insert(0, R)
            
        return returns
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        # log_prob = dist.log_prob(action)

        return action.numpy()
        
    def train(self, next_state):
        returns = self.compute_rewards_to_go()
        returns   = torch.cat(returns).detach()
        
        probs = self.actor.pi(torch.FloatTensor(self.buffer.states), softmax_dim=1)
        probs = probs.gather(1, torch.IntTensor(self.buffer.actions).long())
        log_probs = torch.log(probs)[:, 0]
        
        actor_loss  = -(log_probs * returns.detach()).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        self.buffer.reset()
    