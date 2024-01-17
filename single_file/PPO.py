import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 100
h_size = 300

NN_LAYER_1 = 400
NN_LAYER_2 = 300



class SingleActor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(SingleActor, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc_pi = nn.Linear(NN_LAYER_1, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        probs = F.softmax(x, dim=softmax_dim)
        
        return probs
        
    
class SingleVNet(nn.Module):
    def __init__(self, obs_space):
        super(SingleVNet, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc_v  = nn.Linear(NN_LAYER_1, 1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    

class OnPolicyBuffer:
    def __init__(self, state_dim, length):
        self.state_dim = state_dim
        self.length = length
        self.reset()
        
        self.ptr = 0
        
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.done_masks[self.ptr] = 1 - done

        self.ptr += 1
    
    def reset(self):
        self.states = np.zeros((self.length, self.state_dim))
        self.actions = np.zeros((self.length, 1), dtype=np.int64)
        self.next_states = np.zeros((self.length, self.state_dim))
        self.rewards = np.zeros((self.length, 1))
        self.done_masks = np.zeros((self.length, 1))
        
        self.ptr = 0
   
    def make_data_batch(self):
        states = torch.FloatTensor(self.states[0:self.ptr])
        actions = torch.LongTensor(self.actions[0:self.ptr])
        next_states = torch.FloatTensor(self.next_states[0:self.ptr])
        rewards = torch.FloatTensor(self.rewards[0:self.ptr])
        dones = torch.FloatTensor(self.done_masks[0:self.ptr])
        
        self.reset()
        
        return states, actions, next_states, rewards, dones



class PPO:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.network = None
        self.optimizer = None

        self.actor = SingleActor(self.state_dim, self.action_dim)
        self.critic = SingleVNet(self.state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
        self.replay_buffer = OnPolicyBuffer(state_dim, 10000)
        
    def act(self, obs):
        prob = self.actor.pi(torch.from_numpy(obs).float())
        m = torch.distributions.Categorical(prob)
        a = m.sample().item()

        return a
    
    def generalised_advantage_estimation(self, delta):
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.FloatTensor(advantage_lst)
            
        return advantage
            
    def train(self, next_state=None):
        if self.replay_buffer.ptr < T_horizon:
            return

        states, actions, next_states, rewards, done_masks = self.replay_buffer.make_data_batch()

        for i in range(K_epoch):
            td_target = rewards + gamma * self.critic.v(next_states) * done_masks
            delta = td_target - self.critic.v(states)
            delta = delta.detach().numpy()

            advantage = self.generalised_advantage_estimation(delta)

            probs = self.actor.pi(states, softmax_dim=1)
            probs_for_actions = probs.gather(1,actions)
            # cloning and calling detatch() is how the surrogate objective calculated. Calling detach() on a tensor removes it from the gradient calculation
            prob_a = probs_for_actions.clone().detach()
            ratio = torch.exp(torch.log(probs_for_actions) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            # implementing the clipped surrogate objective
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surrogate_1, surrogate_2) + F.smooth_l1_loss(self.critic.v(states) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.clf()
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001) 

def plot_final(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.clf()
    plt.title("PPO")
    plt.plot(rewards)
    plt.savefig("images/ppo.png")
    
def OnPolicyTrainingLoop_eps(agent, env, batch_eps=1, view=False):
    frame_idx    = 0
    training_rewards = []
    cum_lengths = []
    ep_reward = 0

    while frame_idx < 50000:
        for ep in range(batch_eps):
            (state, _), done = env.reset(), False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.replay_buffer.add(state, action, next_state, reward/100, done)
        
                ep_reward += reward
                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0 and view:
                    plot(frame_idx, training_rewards)
        
            print(f"{frame_idx} -> Episode reward: ", ep_reward)
            training_rewards.append(ep_reward)
            cum_lengths.append(frame_idx)
            ep_reward = 0
    
        agent.train()

    plot_final(frame_idx, training_rewards)

    return cum_lengths, training_rewards


def test_ppo(): 
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PPO(state_dim, n_acts)
    
    OnPolicyTrainingLoop_eps(agent, env, 5, view=True)
    
    
if __name__ == "__main__":
    test_ppo()