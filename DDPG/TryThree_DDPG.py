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

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


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


class ModelDDPG:
    def __init__(self, num_actions, action_noise):
        self.action_noise = action_noise

        self.q = CriticModelQ()
        self.mu = ActorModel(num_actions)

        self.q_p = CriticModelQ()
        self.mu_p = ActorModel(num_actions)

        self.q_p.set_weights(self.q.get_weights())
        self.mu_p.set_weights(self.mu.get_weights())

    def train(self, batch):
        gamma = 0.99
        obs1, obs2 = batch['obs1'], batch['obs2']
        actions, rewards, dones = batch['acts'], batch['rews'], batch['done']

        # possibly clip observations by vlue (-5,5) and normalise
        acts2 = self.mu_p(obs2)
        q_obs2 = self.q_p(obs2, acts2)

        target_q = rewards + (1-dones) * gamma * q_obs2

        actor_grads, actor_loss = self.get_actor_grads(obs1)
        critic_grads, critic_loss = self.get_critic_grads(obs1, actions, target_q)

        self.mu.opt.apply_gradients(zip(actor_grads, self.mu.trainable_variables))
        self.q.opt.apply_gradients(zip(critic_grads, self.q.trainable_variables))

        return critic_loss, actor_loss

    def get_actor_grads(self, obs1):
        with tf.GradientTape() as tape:
            actions = self.mu(obs1)
            critic_vals = self.q(obs1, actions)
            actor_loss = - tf.reduce_mean(critic_vals) # maximise the q values

        actor_grads = tape.gradient(actor_loss, self.mu.trainable_variables)

        return actor_grads, actor_loss
        
    def get_critic_grads(self, obs1, actions, target_q):
        with tf.GradientTape() as tape:
            critic_vals = self.q(obs1, actions)
            critic_target_vals = target_q
            critic_loss = tf.reduce_mean(tf.square(critic_vals - critic_target_vals))

        critic_grads = tape.gradient(critic_loss, self.q.trainable_variables)

        return critic_grads, critic_loss

    def update_networks(self):
        for var, target_var in zip(self.mu.variables, self.mu_p.variables):
            target_var.assign((1. - tau) * target_var + tau * var)
        for var, target_var in zip(self.q.variables, self.q_p.variables):
            target_var.assign((1. - tau) * target_var + tau * var)

    def step(self, obs, noise=True):
        clipped_obs = tf.clip_by_value(obs, -5, 5)
        action = self.mu(clipped_obs[None, :]).numpy()
        
        
        if noise:
            act_noise = self.action_noise()
            action += act_noise

        action = tf.clip_by_value(action, -1, 1)
        action = tf.squeeze(action, axis=0)
        return action.numpy()



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

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])



def learn():
    # env = gym.make('HalfCheetah-v2')
    env = gym.make('Pendulum-v0')

    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]

    mu = np.zeros(act_shape)
    stddev = 0.2
    sigma = stddev * np.ones(act_shape)
    action_noise = OrnsteinUhlenbeckActionNoise(mu, sigma)

    buffer = ReplayBuffer(obs_shape, act_shape, 10000)
    model = ModelDDPG(act_shape, action_noise)

    max_action = env.action_space.high
    print(f"Max action: {max_action}")

    ep_rewards = []
    for epoch in range(500):
        action_noise.reset()
        state = env.reset()
        episode_reward, done = 0, False
        while not done: #max steps
            action = model.step(state)

            new_state, reward, done, _ = env.step(max_action * action)
            buffer.store(state, action, reward, new_state, done)
            episode_reward += reward

            state = new_state

        for _ in range(10): # train steps
            batch = buffer.sample_batch()
            cl, al = model.train(batch) 
            model.update_networks()
        ep_rewards.append(episode_reward)

        print(f"EP: {epoch} -> Reward: {episode_reward} -> cl, al: {cl} ; {al}")
        lib.plot(ep_rewards)
        # possibly add eval here 

        










if __name__ == "__main__":
    learn()