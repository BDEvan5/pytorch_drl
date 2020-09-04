import numpy as np
import tensorflow as tf
import gym
import LibUtils as lib
import collections 
import random 

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
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst
        # return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
        #        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
        #        torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class NetQ(tf.keras.Model):
    def __init__(self):
        super(NetQ, self).__init__()

        self.fc_a = tf.keras.layers.Dense(64, activation='relu')
        self.fc_s = tf.keras.layers.Dense(64, activation='relu')

        self.fc_q = tf.keras.layers.Dense(32, activation='relu')
        self.fc_3 = tf.keras.layers.Dense(1)

    def call(self, s, a):
        s_tensor = tf.convert_to_tensor(s)
        s_tensor = tf.squeeze(s_tensor, axis=1)
        a_tensor = tf.convert_to_tensor(a)

        h1 = self.fc_s(s_tensor)
        h2 = self.fc_a(a_tensor)

        cat = tf.concat([h1, h2], axis=1)
        q = self.fc_q(cat)
        v = self.fc_3(q)
        # value = tf.squeeze(v, axis=0)

        return v

class NetMu(tf.keras.Model):
    def __init__(self, num_actions):
        super(NetMu, self).__init__()

        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc_mu = tf.keras.layers.Dense(num_actions, activation='tanh')

    def call(self, s):
        s_tensor = tf.convert_to_tensor(s)

        x = self.fc1(s_tensor)
        y = self.fc2(x)
        mu = self.fc_mu(y)

        mu_ret = tf.squeeze(mu, axis=-1)

        return mu_ret

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

def train(mu, mu_target, q, q_target, memory, q_opt, mu_opt):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    target = r + gamma * q_target(s_prime, mu_target(s_prime))
    
    huber = tf.keras.losses.Huber()
    with tf.GradientTape() as tape:
        detached_target = tf.stop_gradient(target)
        q_loss = huber(q(s,a), detached_target)
    q_grads = tape.gradient(q_loss, q.trainable_variables)
    q_opt.apply_gradients(zip(q_grads, q.trainable_variables))

    with tf.GradientTape() as tape:
        mu_loss = - tf.reduce_mean(q(s, mu(s)))
    mu_grads = tape.gradient(mu_loss, mu.trainable_variables)
    mu_opt.apply_gradients(zip(mu_grads, mu.trainable_variables))

def soft_update(net, net_target):
    for var, target_var in zip(net.trainable_variables, net_target.trainable_variables):
        new_val = (1. - tau) * target_var + tau * var
        target_var.assign(new_val)

def main():
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer()

    n_acts = env.action_space.shape[0]
    q, q_target = NetQ(), NetQ()
    mu, mu_target = NetMu(n_acts), NetMu(n_acts)
    q.set_weights(q_target.get_weights())
    mu.set_weights(mu_target.get_weights())

    mu_opt = tf.keras.optimizers.Adam(lr=lr_mu)
    q_opt = tf.keras.optimizers.Adam(lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    max_act = env.action_space.high[0]
    plot_rewards = []

    score = 0.0
    print_interval = 10

    for n_epi in range(1000):
        s = env.reset()[None, :]
        ep_reward = 0
        done = False

        while not done:
            a = mu(s).numpy()[0]
            a = a * max_act + ou_noise()[0]
            s_prime, r, done,_ = env.step([a])
            s_prime = s_prime[None, :]
            memory.put((s, a, r/100, s_prime, done))
            score += r
            ep_reward += r
            s = s_prime

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_opt, mu_opt)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 1:
            print(f"{n_epi} :--> score: {score} ")
            score = 0.0

        plot_rewards.append(ep_reward)
        lib.plot(plot_rewards)


if __name__ == "__main__":
    main()
        





