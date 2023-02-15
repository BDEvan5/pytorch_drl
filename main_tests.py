from Components.Algorithms.ddpg import DDPG
from Components.Algorithms.sac import SAC
from Components.Algorithms.td3 import TD3
from Components.Algorithms.a2c import A2C
from Components.Algorithms.dqn import DQN
from Components.Algorithms.a2c_ent import A2C_ent
from Components.Algorithms.ppo import PPO
from Components.TrainingLoops import OffPolicyTrainingLoop, observe, OnPolicyTrainingLoop

import matplotlib.pyplot as plt
import numpy as np
import gym 
import utils




def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    OffPolicyTrainingLoop(agent, env)
    
    
def test_td3():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    
    observe(env, agent.memory, 10000)
    OffPolicyTrainingLoop(agent, env)
        
def test_sac():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env = utils.NormalizedActions(env)
    
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    
    observe(env, agent.memory, 10000)
    OffPolicyTrainingLoop(agent, env)
    
    
def test_a2c():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C(state_dim, n_acts, num_steps)
    
    OnPolicyTrainingLoop(agent, env, num_steps)
     
def test_dqn():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = DQN(state_dim, n_acts)
    
    OffPolicyTrainingLoop(agent, env)
     
def test_a2c_ent():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C_ent(state_dim, n_acts, num_steps)
    
    OnPolicyTrainingLoop(agent, env, num_steps)
       
def test_ppo(): 
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PPO(state_dim, n_acts, num_steps)
    
    OnPolicyTrainingLoop(agent, env, num_steps)
    

    
    
    
if __name__ == '__main__':
    # test_ddpg()
    # test_td3()
    # test_sac()
    
    test_dqn()
    # test_a2c()
    # test_ppo()
    # test_a2c_ent()
    
        