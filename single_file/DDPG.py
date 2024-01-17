import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
BATCH_SIZE   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update
MEMORY_SIZE = 100000


NN_LAYER_1 = 400
NN_LAYER_2 = 300

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q
    
    
class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim, action_scale):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

        self.action_scale = action_scale

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * self.action_scale
        return mu


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

class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr
   

class DDPG:
    def __init__(self, state_dim, action_dim, action_scale):
        self.action_scale = action_scale
        
        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)

        self.critic = DoubleQNet(state_dim, action_dim)
        self.critic_target = DoubleQNet(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = DoublePolicyNet(state_dim, action_dim, action_scale)
        self.actor_target = DoublePolicyNet(state_dim, action_dim, action_scale)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.mu_optimizer = optim.Adam(self.actor.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.critic.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def act(self, state):
        action = self.actor(torch.from_numpy(state).float()) * self.action_scale
        action = action.detach().numpy() + self.ou_noise()
        
        return action
      
    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE: return
        
        states, actions, next_states, rewards, done_masks  = self.replay_buffer.sample(BATCH_SIZE)
        
        target = rewards + gamma * self.critic_target(next_states, self.actor_target(next_states)) * done_masks
        q_loss = F.smooth_l1_loss(self.critic(states,actions), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        mu_loss = -self.critic(states,self.actor(states)).mean() 
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()
    
        soft_update(self.critic, self.critic_target, tau)
        soft_update(self.actor, self.actor_target, tau)
     
    
        
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    

def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001) 

def OffPolicyTrainingLoop(agent, env, training_steps=10000, view=True):
    lengths, rewards = [], []
    (state, _), done = env.reset(), False
    ep_score, ep_steps = 0, 0
    for t in range(1, training_steps):
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        
        done = 0 if ep_steps + 1 == 200 else float(done)
        agent.replay_buffer.add(state, action, next_state, reward, done)  
        ep_score += reward
        ep_steps += 1
        state = next_state
        
        agent.train()
        
        if done:
            lengths.append(ep_steps)
            rewards.append(ep_score)
            (state, _), done = env.reset(), False
            print("Step: {}, Episode :{}, Score : {:.1f}".format(t, len(lengths), ep_score))
            ep_score, ep_steps = 0, 0
        
        
        if t % 1000 == 0 and view:
            plot(t, rewards)
        
    return lengths, rewards


def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    OffPolicyTrainingLoop(agent, env, 10000)
    

if __name__ == '__main__':
    test_ddpg()