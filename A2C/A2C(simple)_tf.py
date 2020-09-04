import numpy as np
import tensorflow as tf
import gym
from matplotlib import pyplot as plt
import LibUtils as lib


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


class VanillaModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.hidden_values = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)

        self.hidden_logs = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions)

        self.opti = tf.keras.optimizers.RMSprop(lr=7e-3)

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        hidden_logs = self.hidden_logs(x)
        logits = self.logits(hidden_logs)

        hidden_values = self.hidden_values(x)
        value = self.value(hidden_values)

        return logits, value

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)

        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        value = tf.squeeze(value, axis=-1)
        action = tf.squeeze(action, axis=-1)

        return action.numpy(), value


class Vanilla:
    def __init__(self, env):
        self.env = env
        self.model = VanillaModel(env.action_space.n)
        
        self.ep_rewards = [0.0]
        self.state = self.env.reset()
        self.buffer = None

    def run_batch(self):
        b = BufferVanilla()
        nsteps = 64
        env, model = self.env, self.model
        state = self.state
        while len(b) <= nsteps:
            action, value = model.action_value(state[None, :])
            next_state, reward, done, _ = env.step(action)
            self.ep_rewards[-1] += reward

            b.add(state, action, value, reward, done)

            if done:
                lib.plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                next_state = env.reset()
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

            state = next_state

        self.state = next_state
        _, q_val = self.model.action_value(state[None, :])
        b.last_q_val = q_val
        
        return b

    def _loss_function(self):
        buffer, q_val = self.buffer, self.q_val
        gamma = 0.99
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            q_vals[len(buffer)-1 - i] = q_val

        advs = q_vals - buffer.values

        acts = np.array(buffer.actions)[:, None]

        obs = tf.convert_to_tensor(buffer.states)
        logits, values = self.model(obs) 

        # value
        value_c = 0.5
        value_loss = value_c * tf.keras.losses.mean_squared_error(q_vals, values)
        value_loss = tf.reduce_mean(value_loss)

        # logits
        entropy_c = 1e-4
        acts = tf.cast(acts, tf.int32)

        weighted_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_ce(acts, logits, sample_weight=advs) # A2C
        # policy_loss = weighted_ce(acts, logits, sample_weight=q_vals) # REINFORCE

        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

        logits_loss = policy_loss - entropy_c * entropy_loss

        total_loss = value_loss + logits_loss
        return total_loss

    def update_model(self, buffer):
        self.buffer = buffer 
        self.buffer = buffer
        self.q_val = buffer.last_q_val

        variables = self.model.trainable_variables
        self.model.opti.minimize(loss = self._loss_function, var_list=variables)


def learn(off_policy=True):
    print("Running Vanilla")
    replay_ratio = 8

    env = gym.make('CartPole-v1')
    replay_buffer = ReplayBuffer(20)

    agent = Vanilla(env)
    for _ in range(200):
        b = agent.run_batch()
        agent.update_model(b)

        if off_policy:
            replay_buffer.add_batch(b)
            for _ in range(replay_ratio):
                b = replay_buffer.get_random_batch()
                agent.update_model(b)


if __name__ == "__main__":
    learn(True)
