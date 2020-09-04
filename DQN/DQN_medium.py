
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

SCORES_CSV_PATH = "./scores.csv"
SCORES_PNG_PATH = "./scores.png"
SOLVED_CSV_PATH = "./solved.csv"
SOLVED_PNG_PATH = "./solved.png"
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100


class ScoreLogger:

    def __init__(self, env_name):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def add_score(self, score, run):
        self._save_csv(SCORES_CSV_PATH, score)
        self._save_png(input_path=SCORES_CSV_PATH,
                       output_path=SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run-CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=SOLVED_CSV_PATH,
                           output_path=SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()