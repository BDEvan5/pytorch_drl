import gym 
import matplotlib.pyplot as plt

max_frames = 50000

def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001)

def DiscreteTrainingLoop(agent, env, num_steps):
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    frame_idx    = 0
    state = env.reset()
    training_rewards = []
    ep_reward = 0

    while frame_idx < max_frames:
        for _ in range(num_steps):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
    
            agent.buffer.add(state, action, next_state, reward/100, done)
            ep_reward += reward
        
            state = next_state
            frame_idx += 1
        
            if done:
                print(f"{frame_idx} -> Episode reward: ", ep_reward)
                training_rewards.append(ep_reward)
                ep_reward = 0
                state = env.reset()
                
            if frame_idx % 1000 == 0:
                plot(frame_idx, training_rewards)
            
        agent.train(next_state)
