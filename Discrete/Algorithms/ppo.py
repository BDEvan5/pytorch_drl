import torch
import torch.optim as optim

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
    def __init__(self, obs_space, action_space, name):
        self.name = name
        self.action_space = action_space
        self.obs_space = obs_space
        
        self.data = []
        self.network = None
        self.optimizer = None

    def create_agent(self, h_size):
        self.actor = Actor(self.obs_space, self.action_space, h_size)
        self.critic = Critic(self.obs_space, self.action_space, h_size)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
    def put_action_data(self, s, a, s_prime, r, done):
        prob = self.actor.pi(torch.from_numpy(s).float())
        transition = (s, a, r, s_prime, prob[a].item(), done)
        
        self.data.append(transition)
        
    def act(self, obs):
        prob = self.actor.pi(torch.from_numpy(obs).float())
        m = torch.distributions.Categorical(prob)
        a = m.sample().item()

        return a

    def train(self):
        if len(self.data) < T_horizon:
            return

        s, a, r, s_prime, done_mask, prob_a = make_data_batch(self.data)
        self.data = []

        for i in range(K_epoch):
            td_target = r + gamma * self.critic.v(s_prime) * done_mask
            delta = td_target - self.critic.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.actor.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
