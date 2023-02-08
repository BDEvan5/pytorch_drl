from Components.LearningAlgs.ddpg import DDPG
from Components.TrainingLoops import ContinuousTrainingLoop
import gym 


def test_ddpg():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 2)
    
    ContinuousTrainingLoop(agent, env)
    
    
    
    
    
    
    
if __name__ == '__main__':
    test_ddpg()
        
        