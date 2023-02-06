import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v1 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
class DDPG:
    def __init__(self, input_dim, output_dim):
        self.memory = ReplayBuffer()

        self.q, self.q_target = QNet(), QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(), MuNet()
        self.mu_target.load_state_dict(self.mu.state_dict())


        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def act(self, state):
        action = self.mu(torch.from_numpy(state).float()) 
        action = action.item() + self.ou_noise()[0]
        
        return action
        
      
    def train(self):
        if self.memory.size() < 1000:
            return
        
        for i in range(10):
            s,a,r,s_prime,done_mask  = self.memory.sample(batch_size)
            
            target = r + gamma * self.q_target(s_prime, self.mu_target(s_prime)) * done_mask
            q_loss = F.smooth_l1_loss(self.q(s,a), target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            mu_loss = -self.q(s,self.mu(s)).mean() # That's all for the policy loss.
            self.mu_optimizer.zero_grad()
            mu_loss.backward()
            self.mu_optimizer.step()
        
            soft_update(self.q, self.q_target)
            soft_update(self.mu, self.mu_target)
     
        
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
    env = gym.make('Pendulum-v1')
    
    score = 0.0
    print_interval = 20
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done:
            a = agent.act(s)
            s_prime, r, done, info = env.step([a])
            agent.memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime
                
        agent.train()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()