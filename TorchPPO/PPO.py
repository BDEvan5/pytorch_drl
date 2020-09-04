import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import gym
import numpy as np 
import LibUtils as lib 

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, num_actions=2, state_space=4):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(state_space, 256)
        self.fc_pi = nn.Linear(256, num_actions)
        self.fc_v = nn.Linear(256, 1)

        self.data = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def v(self, inputs):
        x = F.relu(self.fc1(inputs))
        v = self.fc_v(x)

        return v 

    def pi(self, inputs, softmax_dim=0):
        x = F.relu(self.fc1(inputs))
        pi = F.softmax(self.fc_pi(x), dim=softmax_dim)

        return pi

    def put_data(self, transition):
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

    def train_net(self):
        s, a, r, s_p, done, prob_a = self.make_batch()

        for i in range(K_epoch):
            v_pr = self.v(s_p)
            td_target = gamma * v_pr * done + r
            delta = (td_target - self.v(s)).detach().numpy()

            advantages = []
            advan = 0.0 
            for delta_t in delta[::-1]:
                advan = gamma * lmbda * advan + delta_t
                advantages.append([advan])
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  
            # ratio = torch.div(pi_a, prob)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

            # A2C
            # loss = -torch.log(pi_a) * advantages  + F.smooth_l1_loss(self.v(s), td_target.detach())
            # PPO
            loss = - torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    ep_scores = []
    
    print_interval = 1
    s, done, score = env.reset(), False, 0.0
    for n_epi in range(1000):
        for t in range(T_horizon):
            prob = model.pi(torch.from_numpy(s).float())
            m = torch.distributions.Categorical(prob)
            a = m.sample().item()
            s_p, r, done, _ = env.step(a)

            model.put_data((s, a, r/100.0, s_p, prob[a].item(), done))
            s = s_p
            score += r 
            if done:
                ep_scores.append(score)
                lib.plot(ep_scores)
                print(f"{len(ep_scores)}: --> Score: {score}")
                s, done, score = env.reset(), False, 0.0
                break
        model.train_net()  

if __name__ == "__main__":
    main()
