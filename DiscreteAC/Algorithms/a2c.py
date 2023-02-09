import torch
import torch.optim as optim

from DiscreteAC.Networks import Actor, Critic
from DiscreteAC.OnPolicyBuffer import OnPolicyBuffer

lr          = 3e-4

class A2C:
    def __init__(self, state_dim, n_acts, num_steps) -> None:
        self.actor = Actor(state_dim, n_acts)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buffer = OnPolicyBuffer(state_dim, num_steps)
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.numpy()
        
            
    def compute_rewards_to_go(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.buffer.rewards))):
            R = torch.FloatTensor(self.buffer.rewards[step]) + gamma * R * torch.FloatTensor(self.buffer.done_masks[step])
            returns.insert(0, R)
            
        return returns
        
    def train(self, next_state):
        next_state = torch.FloatTensor(next_state)
        next_value = self.critic.v(next_state)
        returns = self.compute_rewards_to_go(next_value)

        states = torch.FloatTensor(self.buffer.states)
        actions = torch.IntTensor(self.buffer.actions)
        # actions = torch.cas
        
        probs = self.actor.pi(states, softmax_dim=1)
        probs = probs.gather(1, actions.long())
        log_probs = torch.log(probs)
        # log_probs = torch.zeros_like(actions)
        # for i in range(len(self.buffer.actions)):
        #     dist = self.actor.pi(states[i])
        #     log_probs[i] = dist.log_prob(actions[i])
        # log_probs = dists.log_prob(actions)
        # log_probs = log_probs[:, 0]
        values    = self.critic.v(states)

        returns   = torch.cat(returns).detach()
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.buffer.reset()
    
    