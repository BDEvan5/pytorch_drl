
class DQN:
    def __init__(self, obs_space, action_space, name="Agent"):
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory = SmartBufferDQN(obs_space)

        self.name = name
        self.model = None 
        self.target = None
        self.optimizer = None

        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

    def create_agent(self, h_size):
        obs_space = self.obs_space
        action_space = self.action_space

        self.model = Qnet(obs_space, action_space, h_size)
        self.target = Qnet(obs_space, action_space, h_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def sample_action(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def experience_replay(self):
        n_train = 1
        for i in range(n_train):
            if self.memory.size() < BATCH_SIZE:
                return
            s, a, s_p, r, done = self.memory.sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

