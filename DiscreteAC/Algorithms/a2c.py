import torch
import torch.optim as optim

from DiscreteAC.Networks import Actor, Critic
from DiscreteAC.OnPolicyBuffer import OnPolicyBuffer

lr          = 3e-4

class A2C:
    def __init__(self, num_inputs, num_outputs) -> None:
        self.actor = Actor(num_inputs, num_outputs)
        self.critic = Critic(num_inputs)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buffer = OnPolicyBuffer()
        
    def act(self, state):
        state = torch.FloatTensor(state)
        
        dist = self.actor.pi(state)
        value = self.critic.v(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy(), log_prob, value
        
    def train(self, next_state):
        next_state = torch.FloatTensor(next_state)
        next_value = self.critic.v(next_state)
        returns = self.buffer.compute_rewards_to_go(next_value)

        log_probs = torch.stack(self.buffer.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.buffer.values)
        
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.buffer.reset()
    
    