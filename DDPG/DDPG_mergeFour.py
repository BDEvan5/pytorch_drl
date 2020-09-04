import numpy as np
import tensorflow as tf
import gym
import LibUtils as lib

import numpy as np
import tensorflow as tf
import gym
import LibUtils as lib

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self):
        return self.size


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
        action = self.output_logs(x)
        action = tf.squeeze(action, axis=-1)

        return action

    
class CriticModelQ(tf.keras.Model):
    def __init__(self):
        super(CriticModelQ, self).__init__()
        # input is obstervation + action dimension
        self.fc_a = tf.keras.layers.Dense(64, activation='relu')
        self.fc_s = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_values3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_values = tf.keras.layers.Dense(1, activation='linear')

        self.opt = tf.keras.optimizers.Adam(lr=lr_q)

    def call(self, obs, actions):
        o = tf.convert_to_tensor(obs)
        a = tf.convert_to_tensor(actions)

        o1 = self.fc_s(o)
        a1 = self.fc_a(a)
        x = tf.concat([o1, a1], axis=1)

        x = self.hidden_values3(x)
        value = self.output_values(x)

        return value

def train(mu, mu_p, q, q_p, memory):
    batch = memory.sample_batch()
    obs1, obs2 = batch['obs1'], batch['obs2']
    actions, rewards, dones = batch['acts'], batch['rews'], batch['done']

    target_q = rewards + gamma * q_p(obs2, mu_p(obs2)[:, None]) # * (1-dones)

    huber = tf.keras.losses.Huber()
    with tf.GradientTape() as tape:
        q_loss = huber(tf.stop_gradient(target_q), q(obs1, actions))
        
    q_grads = tape.gradient(q_loss, q.trainable_variables)
    q.opt.apply_gradients(zip(q_grads, q.trainable_variables))

    with tf.GradientTape() as tape:
        mu_loss = - tf.reduce_mean(q(obs1, mu(obs1)[:, None]))

    mu_grads = tape.gradient(mu_loss, mu.trainable_variables)
    mu.opt.apply_gradients(zip(mu_grads, mu.trainable_variables))

def soft_update(net, net_target):
    for var, target_var in zip(net.trainable_variables, net_target.trainable_variables):
        target_var.assign((1. - tau) * target_var + tau * var)

def main():
    env = gym.make('Pendulum-v0')
    
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]
    max_act = env.action_space.high
    noise_mu = np.zeros(act_shape)

    memory = ReplayBuffer(obs_shape, act_shape, buffer_limit)
    q, q_target = CriticModelQ(), CriticModelQ()
    q.set_weights(q_target.get_weights())
    mu, mu_target = ActorModel(act_shape), ActorModel(act_shape)
    mu.set_weights(mu_target.get_weights())
    ou_noise = OrnsteinUhlenbeckNoise(mu=noise_mu)

    ep_rewards = []

    for ep in range(1000):
        episode_reward, done, s = 0.1, False, env.reset()

        while not done:
            a = mu(s[None, :])
            a = a.numpy() + ou_noise()[0]
            
            s_p, r, done, _ = env.step(a * max_act)
            memory.store(s, a, r/100, s_p, done)
            episode_reward += r

            s = s_p

        if len(memory) > 2000:
            for _ in range(10):
                train(mu, mu_target, q, q_target, memory)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)

        ep_rewards.append(episode_reward)
        lib.plot(ep_rewards)
        print(f"EP: {ep} -> Reward: {episode_reward}")



if __name__ == "__main__":
    main()