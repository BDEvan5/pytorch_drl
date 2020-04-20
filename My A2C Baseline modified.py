import tensorflow as tf
import numpy as np
import gym 
from matplotlib import pyplot as plt


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')

        self.hidden_values = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)

        self.hidden_logs = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions)

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



def _value_loss(returns, values):
    # standard mse loss - returns=value to move to, value = what I had
    value_c = 0.5
    loss = value_c * tf.keras.losses.mean_squared_error(returns, values)
    return loss

def _logits_loss(actions_and_advs, logits):
        entropy_c = 1e-4
        print("loss acts advs")
        print(actions_and_advs)
        # print(actions_and_advs.numpy())
        acts, advs = tf.split(actions_and_advs, 2, axis=-1)

        weighted_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        acts = tf.cast(acts, tf.int32)
        # I think the sample weigths scale the loss according to how big the adv is.
        # policy loss is the entropy scaled to adv. 
        # entropy is how different was the choice from what was expected. 
        policy_loss = weighted_ce(acts, logits, sample_weight=advs)

        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

        total_policy_loss = policy_loss - entropy_c * entropy_loss
        return total_policy_loss


def plot(values, moving_avg_period, title, figure_n):
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


class ReplayMem:
    def __init__(self, obs_shape):
        self.batch_sz = 64
        self.obs_size = (self.batch_sz,) + obs_shape

        self.step = 0

        self.actions = np.empty((self.batch_sz,), dtype=np.int32)
        self.rewards = np.empty(self.batch_sz)
        self.dones = np.empty(self.batch_sz)
        self.values = np.empty(self.batch_sz)
        self.observations = np.empty(self.obs_size)

    def add_step(self, action, reward, done, value, observation):
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.observations[self.step] = observation

        self.step +=1

    def is_full(self):
        if self.step < self.batch_sz:
            return 0
        return 1

    def clear_mem(self):
        self.step = 0

        self.actions = np.empty((self.batch_sz,), dtype=np.int32)
        self.rewards = np.empty(self.batch_sz)
        self.dones = np.empty(self.batch_sz)
        self.values = np.empty(self.batch_sz)
        self.observations = np.empty(self.obs_size)


class Agent:
    def __init__(self, model, env, buffer):
        lr = 7e-3
        self.gamma = 0.99

        self.model = model
        self.env = env
        self.buffer = buffer

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                    loss=[_logits_loss, _value_loss])

        self.model.action_value(env.reset()[None, :])
        self.model.summary()

    def train(self, updates=250):
        ep_rewards = [0.0]
        next_obs = self.env.reset()

        for update in range(updates):
            while not self.buffer.is_full():
                obs = next_obs.copy()
                action, value = self.model.action_value(next_obs[None, :])
                next_obs, reward, done, _ = env.step(action)

                self.buffer.add_step(action, reward, done, value, obs)

                ep_rewards[-1] += reward

                if done:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            self._update_network(next_obs)
            plot(ep_rewards, 10, "CartpoleTraining", 2)

        return ep_rewards
            
    def _update_network(self, next_obs):
        _, next_value = self.model.action_value(next_obs[None,:])

        returns, advs = self._returns_advs(next_value)

        acts_advs = np.concatenate([self.buffer.actions[:, None], advs[:, None]], axis=-1)

        o = self.buffer.observations
        losses = self.model.train_on_batch(o, [acts_advs, returns])
        
        self.buffer.clear_mem()

    def _returns_advs(self, next_value):
        returns = np.zeros_like(self.buffer.rewards)
        returns = np.append(returns, next_value, axis=-1) 

        for t in reversed(range(self.buffer.rewards.shape[0])):
            returns[t] = self.buffer.rewards[t] + self.gamma * returns[t + 1] * (1 - self.buffer.dones[t])
        returns = returns[:-1]

        advs = returns - self.buffer.values

        return returns, advs

    def test(self, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward





if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)
    buffer = ReplayMem(env.observation_space.shape)

    agent = Agent(model, env, buffer)

    agent.train()
    print(f"Tested Score: {agent.test()} / 200")