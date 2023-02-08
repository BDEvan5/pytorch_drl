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
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


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


# class Actor(nn.Module):   
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, action_dim)

#         self.max_action = max_action


#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.max_action * torch.tanh(self.l3(x)) 
#         return x

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, 1)

#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim, 400)
#         self.l5 = nn.Linear(400, 300)
#         self.l6 = nn.Linear(300, 1)


#     def forward(self, x, u):
#         xu = torch.cat([x, u], 1)

#         x1 = F.relu(self.l1(xu))
#         x1 = F.relu(self.l2(x1))
#         x1 = self.l3(x1)

#         x2 = F.relu(self.l4(xu))
#         x2 = F.relu(self.l5(x2))
#         x2 = self.l6(x2)
#         return x1, x2

#     def Q1(self, x, u):
#         xu = torch.cat([x, u], 1)

#         x1 = F.relu(self.l1(xu))
#         x1 = F.relu(self.l2(x1))
#         x1 = self.l3(x1)
#         return x1

class ReplayBuffer(object):
    def __init__(self, max_size=1000000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)

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

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, iterations):
        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            # target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2)
            # target_Q = reward + (done * GAMMA * target_Q).detach()
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * GAMMA * target_Q).detach()

            # Get current Q estimates
            # current_Q1, current_Q2 = self.critic(state, action)
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



def observe(env,replay_buffer, observation_steps):
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


def test():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = env.action_space.high[0]
    print(state_dim, action_dim, max_action)

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    rewards = []
    steps = 0
    observe(env, replay_buffer, 10000)
    for episode in range(50):
        score, done, obs, ep_steps = 0, False, env.reset(), 0
        while not done:
            action = agent.select_action(np.array(obs), noise=0.1)

            new_obs, reward, done, _ = env.step(action) 
            done_bool = 0 if ep_steps + 1 == 500 else float(done)
        
            replay_buffer.add((obs, new_obs, action, reward, done_bool))          
            obs = new_obs
            score += reward
            ep_steps += 1
            agent.train(replay_buffer, 2) # number is of itterations
            steps += 1

        rewards.append(score)
        print(f"Ep: {episode} -> score: {score} -> steps {steps}")

if __name__ == "__main__":
    test()
 
    