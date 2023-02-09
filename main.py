from Components.LearningAlgs.ddpg import DDPG
# from Components.LearningAlgs.td3_2 import TD3
from Components.LearningAlgs.td3 import TD3
from Components.TrainingLoops import ContinuousTrainingLoop, observe
import gym 

from DiscreteAC.TrainingLoop import DiscreteTrainingLoop
from DiscreteAC.Algorithms.a2c import A2C


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

    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    agent = A2C(num_inputs=num_inputs, num_outputs=num_outputs)
    
    DiscreteTrainingLoop(agent, env)
    
    
if __name__ == '__main__':
    # test_ddpg()
    # test_td3()
    
    test_a2c()
        
        