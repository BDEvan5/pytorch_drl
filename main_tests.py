
from modular.Algorithms.ddpg import DDPG
from modular.Algorithms.sac import SAC
from modular.Algorithms.td3 import TD3
from modular.Algorithms.pg import PolicyGradient
from modular.Algorithms.a2c import A2C
from modular.Algorithms.dqn import DQN
from modular.Algorithms.ppo import PPO
from modular.TrainingLoops import OffPolicyTrainingLoop, observe, OnPolicyTrainingLoop_eps

import matplotlib.pyplot as plt
import numpy as np
import gym 
import utils


     
def test_dqn():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = DQN(state_dim, n_acts)
    
    OffPolicyTrainingLoop(agent, env, 15000, algorithm="DQN")
     
def test_pg():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PolicyGradient(state_dim, n_acts, num_steps)
    
    
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True, algorithm="Policy Gradient")
     

def test_a2c():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C(state_dim, n_acts)
    
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True, algorithm="A2C")
       
def test_ppo(): 
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PPO(state_dim, n_acts)
    
    OnPolicyTrainingLoop_eps(agent, env, 5, view=True, algorithm="PPO")
    


def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    OffPolicyTrainingLoop(agent, env, 10000, algorithm="DDPG")
    
    
def test_td3():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    
    observe(env, agent.replay_buffer, 10000)
    OffPolicyTrainingLoop(agent, env, 10000, algorithm="TD3")
        
def test_sac():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env = utils.NormalizedActions(env)
    
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    
    observe(env, agent.replay_buffer, 10000)
    OffPolicyTrainingLoop(agent, env, 10000, algorithm="SAC")
    
    
    
    
    
if __name__ == '__main__':
    # test_dqn()
    # test_pg()
    # test_a2c()
    # test_ppo()
    
    # test_ddpg()
    test_td3()
    # test_sac()
    
    
        