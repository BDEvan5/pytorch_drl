import tensorflow as tf 
import gym 
import numpy as np 
import LibUtils as lib 

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(tf.keras.Model):
    def __init__(self, num_actions):
        super(PPO, self).__init__()

        self.fc_pi1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_pi = tf.keras.layers.Dense(num_actions) 
        self.fc_v1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_v = tf.keras.layers.Dense(1)

        self.data = []
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    def v(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.fc_v1(x)
        v = self.fc_v(x)
        v = tf.squeeze(v, axis=-1)
        v = tf.squeeze(v, axis=-1)

        return v # returns (1)

    def pi(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.fc_pi1(x)
        x = self.fc_pi(x)
        prob = tf.nn.softmax(x)

        return prob # returns (1, 2)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        sL, aL, rL, s_pL, probL, doneL = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_p, prob, done = transition
            sL.append(s)
            aL.append(a)
            rL.append(r)
            s_pL.append(s_p)
            probL.append(prob)
            doneL.append(0 if done else 1)
        self.data = []

        return sL, aL, rL, s_pL, probL, doneL

    def train_net(self):
        s, a, r, s_p, prob, done = self.make_batch()

        index = np.arange(len(a))[:, None]
        indices = np.concatenate([index, np.array(a)[:, None]], axis=-1)

        for i in range(K_epoch):
            v_pr = self.v(s_p)
            td_target = gamma * v_pr * done + r
            delta = (td_target - self.v(s)).numpy()

            advantages = []
            advan = 0.0
            for delta_t in delta[::-1]:
                advan = gamma * lmbda * advan + delta_t
                advantages.append(advan)
            advantages.reverse()

            huber = tf.keras.losses.Huber()
            with tf.GradientTape() as tape:
                pi = tf.squeeze(self.pi(s), axis=1)
                pi_a = tf.gather_nd(pi, indices)
                ratio = tf.divide(pi_a, prob)
                
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1-eps_clip, 1+ eps_clip) * advantages
                loss = - tf.minimum(surr1, surr2) 
                loss += huber(self.v(s), tf.stop_gradient(td_target))
            
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

def main():
    env = gym.make('CartPole-v1')
    model = PPO(env.action_space.n)
    score = 0.0
    ep_scores = []
    
    print_interval = 1
    s, done, score = env.reset(), False, 0.0
    for n_epi in range(1000):
        for t in range(T_horizon):
            prob = model.pi(s[None, :])
            a = tf.random.categorical(prob, 1).numpy()[0, 0]
            s_p, r, done, _ = env.step(a)

            model.put_data((s[None, :], a, r/100.0, s_p[None, :], prob[0, a].numpy(), done))
            s = s_p
            score += r 
            if done:
                ep_scores.append(score)
                lib.plot(ep_scores)
                print(f"{len(ep_scores)}: --> Score: {score}")
                s, done, score = env.reset(), False, 0.0
                break
        model.train_net()  

if __name__ == "__main__":
    main()