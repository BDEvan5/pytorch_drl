import torch
import torch.optim as optim
import torch.nn.functional as F

from Discrete.Networks import Actor, Critic
from Discrete.OnPolicyBuffer import OnPolicyBuffer



#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 100


class PPO:
    def __init__(self, state_dim, action_dim, num_steps):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.network = None
        self.optimizer = None

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
        self.buffer = OnPolicyBuffer(state_dim, num_steps*2)
        
    def act(self, obs):
        prob = self.actor.pi(torch.from_numpy(obs).float())
        m = torch.distributions.Categorical(prob)
        a = m.sample().item()

        return a
    
    def calculate_advantage(self, delta):
        #ToDo: I am not fully sure what this is doing.... I think it is combining reward to go and advantages but I must study it.
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
        return advantage
            
    def train(self, _):
        if self.buffer.ptr < T_horizon:
            return

        s, a, s_prime, r, done_mask = self.buffer.make_data_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.critic.v(s_prime) * done_mask
            delta = td_target - self.critic.v(s)
            delta = delta.detach().numpy()

            advantage = self.calculate_advantage(delta)

            pi = self.actor.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            prob_a = pi_a.clone().detach()
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

