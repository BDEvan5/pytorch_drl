import gym
import random
import collections
import numpy as np

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
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
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst
    
    def size(self):
        return len(self.buffer)

class ActorModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorModel, self).__init__()

        self.hidden_logs = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_logs2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_logs = tf.keras.layers.Dense(num_actions, activation='tanh') #, activation='tanh')

        self.opt = tf.keras.optimizers.Adam(lr=lr_mu)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        x = self.hidden_logs(x)
        x = self.hidden_logs2(x)
        logits = self.output_logs(x)

        return logits

    
class CriticModelQ(tf.keras.Model):
    def __init__(self):
        super(CriticModelQ, self).__init__()
        # input is obstervation + action dimension
        self.hidden_values = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_values2 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_values3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_values = tf.keras.layers.Dense(1, activation='linear')

        self.opt = tf.keras.optimizers.Adam(lr=lr_q)

    def call(self, obs, actions):
        x = tf.concat([obs, actions], axis=-1)
        x = tf.convert_to_tensor(x)

        x = self.hidden_values(x)
        x = self.hidden_values2(x)
        x = self.hidden_values3(x)
        value = self.output_values(x)

        return value

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

def train(mu, mu_target, q, q_target, memory):
    s, a, r, s_p, done = memory.sample(batch_size)

    target_acts = mu_target(s_p)
    target = r + gamma * q_target(s_p, target_acts)
    
    q_vals = q(s, a)
    q_vals = tf.cast(q_vals, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    q_loss = tf.reduce_mean(target, q_vals)
    q.opt.mimimize(q_loss)

    mu_loss = - tf.reduce_mean(s, mu(s))
    mu.opt.minimize(mu_loss)

def soft_update(net, net_target):
    for var, target_var in zip(net, net_target):
        target_var.assign((1. - tau) * target_var + tau * var)

def main():
    env = gym.make('Pendulum-v0')

    n_acts = env.action_space.shape[0]
    memory = ReplayBuffer()
    q, q_target = CriticModelQ(), CriticModelQ()
    mu, mu_target = ActorModel(n_acts), ActorModel(n_acts)
    

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    n_show = 50
    n_print = 20
    max_act = env.action_space.high

    for n_ep in range(10000):
        s, done, score = env.reset()[:, None], False, 0.0
        while not done:
            a = mu(s).numpy() + ou_noise()[0]
            s_p, r, done, _ = env.step(a * max_act)
            memory.put((s, a, r/100, s_p, done))
            score += r[0]
            s = s_p

            # if n_ep % n_show == 1:
            #     env.render()

        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
    
        if n_ep % n_print == 1:
            print(f"{n_ep} --> score: {score}")

if __name__ == "__main__":
    main()