import numpy as np
import tensorflow as tf
import gym
import LibUtils as lib
from collections import deque
import random


class ModelDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(ModelDQN, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(24, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

        self.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        x = self.hidden_1(x)
        x = self.hidden_2(x)
        outputs = self.output_layer(x)

        return outputs

class DQN:
    def __init__(self):
        self.explore_rate = 1

        self.memory = deque(maxlen=100000)
        self.model = ModelDQN(2)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.explore_rate:
            return np.random.randint(2)
        q_vals = self.model.predict(state[None, :])
        return np.argmax(q_vals)

    def experience_replay(self):
        batch_sz = 20
        gamma = 0.95
        if len(self.memory) < batch_sz:
            return
        batch = random.sample(self.memory, batch_sz)
        for state, action, reward, next_state, done in batch:
            q_vals = self.model.predict(next_state[None, :])
            q_target = reward + (1-done) * gamma * np.amax(q_vals)
            q_values = self.model.predict(state[None, :])
            q_values[0][action] = q_target
            self.model.fit(state[None, :], q_values, verbose=0)
        if self.explore_rate > 0.01:
            self.explore_rate *= 0.995

def learn():
    env = gym.make('CartPole-v1')
    dqn = DQN()
    for i in range(200):
        state, done, ep_reward = env.reset(), False, 0
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            reward = reward if not done else -10
            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            dqn.experience_replay()
        print(f"Run: {i} --> eps: {dqn.explore_rate} --> reward: {ep_reward}")


if __name__ == "__main__":
    learn()
