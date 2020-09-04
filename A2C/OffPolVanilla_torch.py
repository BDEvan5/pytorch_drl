import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import collections
import random
import gym
import numpy as np 
import LibUtils as lib 
from matplotlib import pyplot as plt



class BufferVanilla():
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.last_q_val = None

    def add(self, state, action, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.states, 
                self.actions,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

class ReplayBuffer:
    def __init__(self, size=5000):
        self.size = 5000
        self.buffer = []
        self.idx = 0

    def add_batch(self, batch):
        if self.idx > self.size:
            self.buffer.pop(0) # remove oldest batch
        self.buffer.append(batch)
        self.idx += 1

    def get_random_batch(self):
        # each time this is called, it will get a random buffer for training.
        rand_idx = np.random.randint(0, self.idx)
        buffer_return = self.buffer[rand_idx]

        return buffer_return


def plot(values, moving_avg_period=10, title="Training", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def get_moving_average(period, values):

    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.hv = nn.Linear(4, 128)
        self.v = nn.Linear(128, 1)

        self.hlog = nn.Linear(4, 128)
        self.logits = nn.Linear(128, 2)

        self.opti = optim.RMSprop(self.parameters(), 7e-3)

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float)
        hlog = F.relu(self.hlog(x))
        logits = self.logits(hlog)
        

        hv = F.relu(self.hv(x))
        v = self.v(hv)

        return logits, v 



class Model:
    def __init__(self):
        self.policy = Policy()

        self.buffer = None
        self.q_val = None

    def get_action_value(self, obs):
        x = torch.tensor(obs, dtype=torch.float)
        logits, v = self.policy(x)

        m = torch.distributions.Categorical(logits=logits)
        a = m.sample().item()

        return a, v

    def update_model(self, buffer):
        q_val = buffer.last_q_val
        gamma = 0.99
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            v = q_val.detach().item()
            q_vals[len(buffer)-1 - i] = v

        q_vals = torch.tensor(q_vals[:, 0], dtype=torch.float)
        values = torch.cat(tuple(buffer.values))
        advs = q_vals - values

        obs = torch.tensor(buffer.states, dtype=torch.float)
        logits, values = self.policy(obs)
        probs = F.softmax(logits, dim=1)

        values = torch.squeeze(values, dim=-1)
        value_loss = 0.5 * F.mse_loss(values, q_vals)

        acts = np.array(buffer.actions)#[:, None]
        acts = torch.tensor(acts, dtype=torch.long)
        # pi_a = probs.gather(1, acts)
        # logit_loss = - advs.detach() * torch.log(pi_a)

        # acts = acts[:, 0]
        logit_loss = F.cross_entropy(probs, acts) * advs.detach()

        logit_loss = logit_loss.mean()

        total_loss = (logit_loss + value_loss).sum()

        self.policy.opti.zero_grad()
        total_loss.backward()
        self.policy.opti.step()


class RunnerVanilla:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.buffer = BufferVanilla()

        self.ep_rewards = [0.0]
        self.state = self.env.reset()

    def run_batch(self):
        b = BufferVanilla()
        nsteps = 64
        env, model = self.env, self.model
        state = self.state
        while len(b) <= nsteps:
            action, value = model.get_action_value(state)
            next_state, reward, done, _ = env.step(action)
            self.ep_rewards[-1] += reward

            b.add(state, action, value, reward, done)

            if done:
                plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                next_state = env.reset()
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

            state = next_state

        self.state = next_state
        _, q_val = self.model.get_action_value(state)
        b.last_q_val = q_val
        
        return b


def learn():
    print("Running Vanilla")
    replay_ratio = 8 #replay ratio of on to off policy learning

    env = gym.make('CartPole-v1')
    model = Model()
    replay_buffer = ReplayBuffer(20)

    runner = RunnerVanilla(env, model)
    for _ in range(200):
        b = runner.run_batch()
        replay_buffer.add_batch(b)
        model.update_model(b)

        for _ in range(replay_ratio):
            b = replay_buffer.get_random_batch()
            model.update_model(b)



if __name__ == "__main__":
    learn()
