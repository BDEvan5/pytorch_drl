from Continuous.LearningAlgs.ddpg import DDPG
# from Components.LearningAlgs.td3_2 import TD3
from Continuous.LearningAlgs.td3 import TD3
from Continuous.TrainingLoops import ContinuousTrainingLoop, observe
import gym 

from Discrete.TrainingLoop import DiscreteTrainingLoop
from Discrete.Algorithms.a2c import A2C
from Discrete.Algorithms.pg import PolicyGradient
from Discrete.Algorithms.a2c_ent import A2C_ent
from Discrete.Algorithms.ppo import PPO

import matplotlib.pyplot as plt
import numpy as np


def compare_discrete_ac():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    
    agent = A2C(state_dim, n_acts, num_steps)
    a2c_lengths, a2c_rewards = DiscreteTrainingLoop(agent, env, num_steps)
    
    agent = PPO(state_dim, n_acts, num_steps)
    ppo_lengths, ppo_rewards = DiscreteTrainingLoop(agent, env, num_steps)
    
    plt.figure(1, figsize=(5,5))
    plt.plot(a2c_lengths, a2c_rewards, label="A2C", color='blue')
    plt.plot(ppo_lengths, ppo_rewards, label="PPO", color='red')
    
    plt.legend()
    plt.grid()
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    
    plt.savefig("a2c_vs_ppo.svg")
    plt.savefig("a2c_vs_ppo.pdf")
    
    plt.show()
    
if __name__ == "__main__":
    compare_discrete_ac()
    