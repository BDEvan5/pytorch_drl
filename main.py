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


def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    ContinuousTrainingLoop(agent, env)
    
    
def test_td3():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    
    observe(env, agent.memory, 10000)
    ContinuousTrainingLoop(agent, env)
    
    
def test_a2c():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C(state_dim, n_acts, num_steps)
    
    DiscreteTrainingLoop(agent, env, num_steps)
     
def test_a2c_ent():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C_ent(state_dim, n_acts, num_steps)
    
    DiscreteTrainingLoop(agent, env, num_steps)
    
def test_pg(): #! not working
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PolicyGradient(state_dim, n_acts, num_steps)
    
    DiscreteTrainingLoop(agent, env, num_steps)
    
        
def test_ppo(): 
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PPO(state_dim, n_acts, num_steps)
    
    DiscreteTrainingLoop(agent, env, num_steps)
    
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
    
    
    
    
if __name__ == '__main__':
    # test_ddpg()
    # test_td3()
    
    # test_a2c()
    # test_a2c_ent()
    # test_pg()
    
    # test_ppo()
        
    compare_discrete_ac()