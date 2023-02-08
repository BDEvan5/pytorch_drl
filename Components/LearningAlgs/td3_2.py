import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import sys

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
# EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2
MEMORY_SIZE = 1000000


hidden_fc1_units = 400
hidden_fc2_units = 300

class PolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim, action_scale):
        super(PolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_fc1_units)
        self.fc2 = nn.Linear(hidden_fc1_units, hidden_fc2_units)
        self.fc_mu = nn.Linear(hidden_fc2_units, act_dim)

        self.action_scale = action_scale

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * self.action_scale
        return mu

class CriticNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(CriticNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, hidden_fc1_units)
        self.fc2 = nn.Linear(hidden_fc1_units, hidden_fc2_units)
        self.fc_out = nn.Linear(hidden_fc2_units, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q


class SmartBufferTD3(object):
    def __init__(self, max_size=MEMORY_SIZE, state_dim=14, act_dim=1):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.ptr = 0

        self.states = np.empty((max_size, state_dim))
        self.actions = np.empty((max_size, act_dim))
        self.next_states = np.empty((max_size, state_dim))
        self.rewards = np.empty((max_size, 1))
        self.dones = np.empty((max_size, 1))

    def add(self, s, a, s_p, r, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = s_p
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.act_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        # states = torch.FloatTensor(states)
        # actions = torch.FloatTensor(actions)
        # next_states = torch.FloatTensor(next_states)
        # rewards = torch.FloatTensor(rewards)
        # dones = torch.FloatTensor(1- dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = PolicyNet(state_dim, action_dim, max_action)
        self.actor_target = PolicyNet(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = CriticNet(state_dim, action_dim)
        self.critic_target_1 = CriticNet(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = CriticNet(state_dim, action_dim)
        self.critic_target_2 = CriticNet(state_dim, action_dim)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.max_action = max_action
        self.act_dim = action_dim

        self.memory = SmartBufferTD3(MEMORY_SIZE, state_dim, action_dim)


    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def train(self, iterations):
        for it in range(iterations):
            x, u, y, r, d = self.memory.sample(BATCH_SIZE)
            # state, action, next_state, reward, done = self.memory.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            noise_action = action.detach().clone()
            noise = noise_action.data.normal_(0, POLICY_NOISE)
            # noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * GAMMA * target_Q).detach()

            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % POLICY_FREQUENCY == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                soft_update(self.critic_1, self.critic_target_1, tau)
                soft_update(self.critic_2, self.critic_target_2, tau)
                soft_update(self.actor, self.actor_target, tau)


        
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def observe(env,replay_buffer, observation_steps):
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add(obs, action, new_obs, reward, done)        

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()
    print("")


def test():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = env.action_space.high[0]

    agent = TD3(state_dim, action_dim, max_action)
    
    observe(env, agent.memory, 10000)
    ContinuousTrainingLoop2(env, agent)

def ContinuousTrainingLoop2(env, agent):
    rewards = []
    steps = 0
    for episode in range(50):
        score, done, obs, ep_steps = 0, False, env.reset(), 0
        while not done:
            action = agent.act(obs, noise=0.1)

            new_obs, reward, done, _ = env.step(action) 
            done_bool = 0 if ep_steps + 1 == 500 else float(done)
        
            agent.memory.add(obs, action, new_obs, reward, done_bool)  
            agent.train(2) 
                  
            obs = new_obs
            score += reward
            ep_steps += 1
            steps += 1

        rewards.append(score)
        print(f"Ep: {episode} -> score: {score} -> steps {steps}")

if __name__ == "__main__":
    test()
 
    