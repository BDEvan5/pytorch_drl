
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 100

class Network(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc_pi = nn.Linear(h_size, action_space)
        self.fc_v  = nn.Linear(h_size,1)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

class PPO:
    def __init__(self, obs_space, action_space, name):
        self.name = name
        self.action_space = action_space
        self.obs_space = obs_space
        
        self.data = []
        self.network = None
        self.optimizer = None

    def create_agent(self, h_size):
        self.network = Network(self.obs_space, self.action_space, h_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
      
    def put_data(self, transition):

        self.data.append(transition)
        
    def put_action_data(self, s, a, s_prime, r, done):
        prob = self.network.pi(torch.from_numpy(s).float())
        transition = (s, a, r, s_prime, prob[a].item(), done)
        
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def act(self, obs):
        prob = self.network.pi(torch.from_numpy(obs).float())
        m = Categorical(prob)
        a = m.sample().item()

        return a

    def train(self):
        if len(self.data) < T_horizon:
            return

        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.network.v(s_prime) * done_mask
            delta = td_target - self.network.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.network.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.network.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def load(self, directory="./saves"):
        filename = self.name

        self.network = torch.load('%s/%s_network.pth' % (directory, filename))

        print(f"Agent Loaded: {filename}")

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.network, '%s/%s_network.pth' % (directory, filename))

        # print(f"Agent Saved: {filename}")

def main():
    env = gym.make('CartPole-v1')
    model = PPO(4, 2, "myfriend")
    model.create_agent(100)
    # model.create_agent(256)
    score = 0.0
    print_interval = 20

    for n_epi in range(1000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                a = model.act(s)
                s_prime, r, done, info = env.step(a)

                model.put_action_data(s, a, s_prime, r/100.0, done)
                s = s_prime

                score += r
                if done:
                    break

            model.train()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()