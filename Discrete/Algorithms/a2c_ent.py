import torch
import torch.optim as optim

from Discrete.Networks import Actor, Critic
from Discrete.OnPolicyBuffer import OnPolicyBuffer

lr          = 3e-4
gamma=0.99


class A2C_ent:
    def __init__(self, state_dim, n_acts, num_steps) -> None:
        self.actor = Actor(state_dim, n_acts)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.entropy = 0
        self.buffer = OnPolicyBuffer(state_dim, num_steps)
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.entropy += dist.entropy().mean()

        return action.numpy()
        
    def compute_rewards_to_go(self, next_value):
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
        values    = self.critic.v(states)
        
        actions = torch.IntTensor(self.buffer.actions)
        probs = self.actor.pi(states, softmax_dim=1)
        probs = probs.gather(1, actions.long())
        log_probs = torch.log(probs)[:, 0]

        returns   = torch.cat(returns).detach()
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.entropy = 0
        self.buffer.reset()
    
    