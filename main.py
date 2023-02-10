import gym 

from Discrete.TrainingLoop import DiscreteTrainingLoop
from Discrete.Algorithms.a2c import A2C
from Discrete.Algorithms.a2c_ent import A2C_ent
import matplotlib.pyplot as plt
import numpy as np
from Discrete.Algorithms.ppo import PPO
    
    
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
    
    # test_a2c()
    # test_a2c_ent()
    
    # test_ppo()
        
    compare_discrete_ac()