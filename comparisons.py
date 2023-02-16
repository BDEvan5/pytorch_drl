from Components.Algorithms.ddpg import DDPG
from Components.Algorithms.td3 import TD3
from Components.Algorithms.sac import SAC

from Components.TrainingLoops import OffPolicyTrainingLoop, OnPolicyTrainingLoop_eps, observe
from Components.Algorithms.a2c import A2C
from Components.Algorithms.ppo import PPO

import utils
import matplotlib.pyplot as plt
import numpy as np
import gym 


def compare_Components_ac():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    
    agent = A2C(state_dim, n_acts)
    a2c_lengths, a2c_rewards = OnPolicyTrainingLoop_eps(agent, env, 1)
    
    agent = PPO(state_dim, n_acts)
    ppo_lengths, ppo_rewards = OnPolicyTrainingLoop_eps(agent, env, 1)
    
    plt.figure(1, figsize=(5,5))
    plt.plot(a2c_lengths, a2c_rewards, label="A2C", color='blue')
    plt.plot(ppo_lengths, ppo_rewards, label="PPO", color='red')
    
    plt.legend()
    plt.grid()
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    
    plt.savefig("Results/Training/a2c_vs_ppo.svg")
    
    plt.show()
    
    
def compare_Components():
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    env = utils.NormalizedActions(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    observation_steps = 5000
    training_steps = 3000

    agent = SAC(state_dim, action_dim)
    observe(env, agent.replay_buffer, observation_steps)
    sac_lengths, sac_rewards = OffPolicyTrainingLoop(agent, env, training_steps, False)
    
    agent = DDPG(state_dim, action_dim, env.action_space.high[0])
    observe(env, agent.replay_buffer, observation_steps)
    ddpg_lengths, ddpg_rewards = OffPolicyTrainingLoop(agent, env, training_steps, False)
    
    agent = TD3(state_dim, action_dim, env.action_space.high[0])
    observe(env, agent.replay_buffer, observation_steps)
    td3_lengths, td3_rewards = OffPolicyTrainingLoop(agent, env, training_steps, False)
            
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
    # compare_Components_ac()
    compare_Components()
    
    