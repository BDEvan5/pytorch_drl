<<<<<<< HEAD:main.py
import gym 

from Discrete.TrainingLoop import DiscreteTrainingLoop
from Discrete.Algorithms.a2c import A2C
from Discrete.Algorithms.a2c_ent import A2C_ent
import matplotlib.pyplot as plt
import numpy as np
from Discrete.Algorithms.ppo import PPO
    
    
def test_a2c():
=======
from Components.Algorithms.ddpg import DDPG
from Components.Algorithms.sac import SAC
from Components.Algorithms.td3 import TD3
from Components.Algorithms.pg import PolicyGradient
from Components.Algorithms.a2c import A2C
from Components.Algorithms.dqn import DQN
from Components.Algorithms.a2c_ent import A2C_ent
from Components.Algorithms.ppo import PPO
from Components.TrainingLoops import OffPolicyTrainingLoop, observe, OnPolicyTrainingLoop, OnPolicyTrainingLoop_eps

import matplotlib.pyplot as plt
import numpy as np
import gym 
import utils


     
def test_dqn():
>>>>>>> devel:main_tests.py
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = DQN(state_dim, n_acts)
    
    OffPolicyTrainingLoop(agent, env, 15000)
     
def test_pg():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PolicyGradient(state_dim, n_acts, num_steps)
    
<<<<<<< HEAD:main.py

=======
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True)


def test_a2c():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C(state_dim, n_acts, num_steps)
>>>>>>> devel:main_tests.py
    
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True)
     

def test_a2c_ent():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    num_steps = 100

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = A2C_ent(state_dim, n_acts, num_steps)
    
    OnPolicyTrainingLoop_eps(agent, env, 1, view=True)
       
def test_ppo(): 
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim  = env.observation_space.shape[0]
    n_acts = env.action_space.n
    agent = PPO(state_dim, n_acts)
    
    OnPolicyTrainingLoop_eps(agent, env, 5, view=True)
    


def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    OffPolicyTrainingLoop(agent, env, 10000)
    
    
def test_td3():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    
    observe(env, agent.replay_buffer, 10000)
    OffPolicyTrainingLoop(agent, env, 10000)
        
def test_sac():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env = utils.NormalizedActions(env)
    
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    
    observe(env, agent.replay_buffer, 10000)
    OffPolicyTrainingLoop(agent, env, 10000)
    
    
    
    
    
if __name__ == '__main__':
<<<<<<< HEAD:main.py
    
    # test_a2c()
    # test_a2c_ent()
=======
    # test_dqn()
    # test_pg()
    # test_a2c()
    # test_a2c_ent()
    # test_ppo()
    
    # test_ddpg()
    # test_td3()
    test_sac()
    
>>>>>>> devel:main_tests.py
    
        