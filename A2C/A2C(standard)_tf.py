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


class Policy(tf.keras.Model):
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

    def get_action_value(self, obs):
        logits, value = self.predict_on_batch(obs)

        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        value = tf.squeeze(value, axis=-1)
        action = tf.squeeze(action, axis=-1)

        return action.numpy(), value


class Model:
    def __init__(self):
        self.policy = Policy(2)

        self.buffer = None
        self.q_val = None

    def update_model(self, buffer):
        self.buffer = buffer
        self.q_val = buffer.last_q_val

        variables = self.policy.trainable_variables
        self.policy.opti.minimize(loss = self._loss_fcn, var_list=variables)

    def _loss_fcn(self):
        buffer, q_val = self.buffer, self.q_val
        gamma = 0.99
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            q_vals[len(buffer)-1 - i] = q_val

        advs = q_vals - buffer.values
        acts = np.array(buffer.actions)[:, None]
        obs = tf.convert_to_tensor(buffer.states)
        logits, values = self.policy(obs) 

        # value
        value_c = 0.5
        value_loss = value_c * tf.keras.losses.mean_squared_error(q_vals, values)
        value_loss = tf.reduce_mean(value_loss)

        # logits
        entropy_c = 0#1e-4
        acts = tf.cast(acts, tf.int32)

        weighted_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_ce(acts, logits, sample_weight=advs)

        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

        logits_loss = policy_loss - entropy_c * entropy_loss

        total_loss = value_loss + logits_loss
        return total_loss

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
            action, value = model.policy.get_action_value(state[None, :])
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
        _, q_val = self.model.policy.get_action_value(state[None, :])
        b.last_q_val = q_val
        
        return b

def learn():
    print("Running Vanilla")
    replay_ratio = 0 #replay ratio of on to off policy learning

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










