
class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc_h1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc_h2 = tf.keras.layers.Dense(32, activation = 'relu')
        self.fc_v = tf.keras.layers.Dense(1)

        self.opt_v = tf.keras.optimizers.RMSprop(lr=7e-3)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        h1 = self.fc_h1(x)
        h2 = self.fc_h2(h1)
        v = self.fc_v(h2)

        return v

class CrashModel:
    def __init__(self):
        self.policy = Policy()
        self.obs_memory = deque()
        self.reward_mem = deque()
        self.full_memory = deque(maxlen=2000)

        self.plot_values = []

    def expect_crash(self, s, t):
        obs = tf.concat([s, t], axis=-1)
        v = self.policy(obs[None, :])[0, 0]
        self.obs_memory.append(obs)

        self.plot_values.append(v.numpy())
        lib.plot(self.plot_values)

        if v.numpy() < 0:
            return True

        return False

    def add_memory_step(self, r, done):
        gamma = 0.99
        self.reward_mem.append(r)
        if done:
            q_val = 0
            self.reward_mem.reverse()
            self.obs_memory.reverse()
            for s, r in zip(self.obs_memory, self.reward_mem):
                q_val = r + gamma * q_val # only happens from done
                self.full_memory.append((s, q_val))
            self.reward_mem.clear()
            self.obs_memory.clear()

    def train(self):
        batch_sz = 32
        if len(self.full_memory) < batch_sz:
            return
        batch = random.sample(self.full_memory, batch_sz)
        states = [b[0] for b in batch]
        q_vals = [b[1] for b in batch]
        with tf.GradientTape() as tape:
            values = self.policy(states)
            value_loss = tf.keras.losses.mean_squared_error(values, q_vals)
        grads = tape.gradient(value_loss, self.policy.trainable_variables)
        self.policy.opt_v.apply_gradients(zip(grads, self.policy.trainable_variables))

    def run_training(self):
        for i in range(10):
            self.train()

