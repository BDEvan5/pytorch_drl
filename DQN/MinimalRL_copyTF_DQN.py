import gym
import collections
import random
import tensorflow as tf 
import numpy as np 

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


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
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst
        # return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
        #        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
        #        torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(tf.keras.Model):
    def __init__(self):
        super(Qnet, self).__init__()
        # self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='linear')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        h1 = self.fc1(x)
        q = self.fc2(h1)

        return q

    def sample_action(self, obs, epsilon):
        out = self(obs)[0]
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return tf.argmax(out).numpy()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done = memory.sample(batch_size)
        index = np.arange(len(a))[:, None]
        indices = np.concatenate([index, np.array(a)[:, None]], axis=-1)

        huber = tf.keras.losses.Huber()

        q_prime = q_target(s_prime)
        q_prime = tf.squeeze(q_prime, axis=1)
        max_q_prime = tf.maximum(q_prime[:, 0], q_prime[:, 0])
        target = r + gamma * max_q_prime * done 
        with tf.GradientTape() as tape:
            q_out = q(s) 
            q_out = tf.squeeze(q_out, axis=1)
            q_a = tf.gather_nd(q_out, indices)
            loss = huber(q_a, target)

        grads = tape.gradient(loss, q.trainable_variables)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))

def main():
    env = gym.make('CartPole-v1')

    q, q_target = Qnet(), Qnet()
    q_target.set_weights(q.get_weights())

    memory = ReplayBuffer()
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    print_interval = 20

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))
        s, done = env.reset()[None, :], False
        _ = q_target(s) # compiles model
        score = 0

        while not done:
            a = q.sample_action(s, epsilon)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0 if done else 1
            s_prime = s_prime[None, :]
            memory.put((s, a, r/100, s_prime, done_mask))
            s = s_prime

            score += r

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval ==1:
            q_target.set_weights(q.get_weights())
            print(f"{n_epi}-> Score: {score}")
        

if __name__ == "__main__":
    main()