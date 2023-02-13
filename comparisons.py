from Continuous.LearningAlgs.ddpg import DDPG
from Continuous.LearningAlgs.td3 import TD3
from Continuous.LearningAlgs.sac import SAC
from Continuous.TrainingLoops import ContinuousTrainingLoop, observe
import gym 

from Discrete.TrainingLoop import DiscreteTrainingLoop
from Discrete.Algorithms.a2c import A2C
from Discrete.Algorithms.a2c_ent import A2C_ent
from Discrete.Algorithms.ppo import PPO

import utils
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
    
    plt.savefig("Results/Training/a2c_vs_ppo.svg")
    # plt.savefig("Results/Training/a2c_vs_ppo.pdf")
    
    plt.show()
    
def compare_continuous():
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    env = utils.NormalizedActions(env)

    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    observe(env, agent.memory, 10000)
    sac_lengths, sac_rewards = ContinuousTrainingLoop(agent, env)
    
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    observe(env, agent.memory, 10000)
    ddpg_lengths, ddpg_rewards = ContinuousTrainingLoop(agent, env)
    
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    observe(env, agent.memory, 10000)
    td3_lengths, td3_rewards = ContinuousTrainingLoop(agent, env)
            
    plt.figure(1, figsize=(5,5))
    plt.plot(ddpg_lengths, ddpg_rewards, label="DDPG", color=utils.colors[0])
    plt.plot(td3_lengths, td3_rewards, label="TD3", color=utils.colors[1])
    plt.plot(sac_lengths, sac_rewards, label="SAC", color=utils.colors[2])
    
    plt.legend()
    plt.grid()
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    
    plt.savefig("Results/Training/compare_continuous.svg")
    
    plt.show()
    
if __name__ == "__main__":
    # compare_discrete_ac()
    compare_continuous()
    
    