import numpy as np
from collections import deque
import random
import gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
import matplotlib.pyplot as plt
from datetime import datetime
from sys import exit

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


class Critic_gen(tf.keras.Model):
    def __init__(self, state_size, actions_size, hidden_layers):
        super(Critic_gen, self).__init__()

        self.fc1 = Dense(1204, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(300, activation='relu')
        self.fc4 = Dense(1)

    def call(self, x):
        obs = concatenate(x, axis=-1)

        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        v =  self.fc4(h3)

        return v

class Actor_gen(tf.keras.Model):
    def __init__(self, state_size, actions_size, hidden_layers,action_mult=1):
        super(Actor_gen, self).__init__()

        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(200, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        self.fc4 = Dense(actions_size, activation='tanh')

        self.max_act = action_mult

    def call(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        act =  self.fc4(h3)
        act = tf.math.multiply(act, self.max_act)
        return act

class DDPGAgent:
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        #Network layers
        actor_layer = [512,200,128]
        critic_layer = [1024,512,300,1]

        # Main network outputs
        self.mu = Actor_gen((3),(1),actor_layer,self.action_max)
        self.q_mu = Critic_gen((3),(1),critic_layer)

        # Target networks
        self.mu_target = Actor_gen((3),(1),actor_layer,self.action_max)
        self.q_mu_target = Critic_gen((3),(1),critic_layer)
      
        # Copying weights in,
        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())
    
        # optimizers
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        self.q_mu_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
  
        self.replay_buffer = BasicBuffer_a(size = buffer_maxlen, obs_dim = self.obs_dim, act_dim = self.action_dim)
        
        self.q_losses = []
        self.mu_losses = []
        
    def get_action(self, s, noise_scale):
        a =  self.mu.predict(s.reshape(1,-1))[0]
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_max, self.action_max)

    def update(self, batch_size):
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)

        # Updating Ze Critic
        with tf.GradientTape() as tape:
          A2 =  self.mu_target(X2)
          q_target = R + self.gamma  * self.q_mu_target([X2,A2])
          qvals = self.q_mu([X,A]) 
          q_loss = tf.reduce_mean((qvals - q_target)**2)
          grads_q = tape.gradient(q_loss,self.q_mu.trainable_variables)
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)


        #Updating ZE Actor
        with tf.GradientTape() as tape2:
          A_mu =  self.mu(X)
          Q_mu = self.q_mu([X,A_mu])
          mu_loss =  -tf.reduce_mean(Q_mu)
          grads_mu = tape2.gradient(mu_loss,self.mu.trainable_variables)
        self.mu_losses.append(mu_loss)
        self.mu_optimizer.apply_gradients(zip(grads_mu, self.mu.trainable_variables))

        # # updating q_mu network
        temp1 = np.array(self.q_mu_target.get_weights())
        temp2 = np.array(self.q_mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.q_mu_target.set_weights(temp3)
      

     # updating mu network
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.mu_target.set_weights(temp3)

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


