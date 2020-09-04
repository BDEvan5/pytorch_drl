import tensorflow as tf 
from tensorflow.keras.layers import Dense

import numpy as np
from collections import deque
import random
import gym



class BasicBuffer_a:
    
    def __init__(self, size, obs_dim, act_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = np.asarray([rew])
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict= dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])
        return (temp_dict['s'],temp_dict['a'],temp_dict['r'].reshape(-1,1),temp_dict['s2'],temp_dict['d'])

class BasicBuffer_b:
    
    def __init__(self, size, obs_dim = None, act_dim = None):
        self.max_size = size
        self.buffer = deque(maxlen=size)
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.size = min(self.size+1,self.max_size)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        # np.random.seed(0)
        batch = np.random.randint(0, len(self.buffer), size=batch_size)
        for experience in batch:
            state, action, reward, next_state, done = self.buffer[experience]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(300, activation='relu')
        self.fc4 = Dense(1)

    def call(self, x, a):
        # a = a[:, None]
        a = tf.reshape(a, (-1, 1))
        obs = tf.keras.layers.concatenate([x, a], axis=-1)
        # obs = tf.concat([x, a], axis=-1)

        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        v = self.fc4(h3)

        v = tf.squeeze(v, axis=-1)
        return v

class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()

        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(200, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        self.fc4 = Dense(1, activation='tanh')

    def call(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        pi = self.fc4(h3)

        pi = tf.squeeze(pi, axis=-1)
        return pi


class DDPGAgent:
    def __init__(self, env, gamma, tau, maxlen, cr_lr, a_lr):
        self.env = env 
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]

        self.gamma = gamma
        self.tau = tau

        self.mu = Actor()
        self.mu_target = Actor()
        self.q_mu = Critic()
        self.q_mu_target = Critic()

        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())

        self.mu_opt = tf.keras.optimizers.Adam(learning_rate=cr_lr)
        self.q_opt = tf.keras.optimizers.Adam(learning_rate=a_lr)

        self.replay_buffer = BasicBuffer_a(size=maxlen ,obs_dim=self.obs_dim, act_dim=self.action_dim)

        self.q_losses = []
        self.mu_losses = []

    def get_action(self, s, noise_scale):
        a = self.mu.predict(s[None, :])
        a *= self.action_max
        a += noise_scale * np.random.randn(self.action_dim)
        a = np.clip(a, -self.action_max, self.action_max)

        return a

    def update(self, batch_size):
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)
        
        Xten = tf.convert_to_tensor(X)

        # critic update
        with tf.GradientTape() as tape:
            A2 =  self.mu_target(X2)
            q_target = R + self.gamma  * self.q_mu_target(X2, A2)
            qvals = self.q_mu(X,A) 
            q_loss = tf.reduce_mean((qvals - q_target)**2)
            grads_q = tape.gradient(q_loss,self.q_mu.trainable_variables)
        self.q_opt.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)

        #Updating ZE Actor
        with tf.GradientTape() as tape2:
            A_mu =  self.mu(X)
            Q_mu = self.q_mu(X,A_mu)
            mu_loss =  -tf.reduce_mean(Q_mu)
            grads_mu = tape2.gradient(mu_loss,self.mu.trainable_variables)
        self.mu_losses.append(mu_loss)
        self.mu_opt.apply_gradients(zip(grads_mu, self.mu.trainable_variables))
        
        temp1 = np.array(self.q_mu_target.get_weights())
        temp2 = np.array(self.q_mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.q_mu_target.set_weights(temp3)
    
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.mu_target.set_weights(temp3)

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y

def trainer(env, agent, max_episodes, max_steps, batch_size, action_noise):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps-1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size)   


            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


env = gym.make("Pendulum-v0")

max_episodes = 20
max_steps = 500
batch_size = 32

gamma = 0.99
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = trainer(env, agent, max_episodes, max_steps, batch_size,action_noise=0.1)
