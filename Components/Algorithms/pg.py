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
    