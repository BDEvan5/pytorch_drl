import sys
from matplotlib import pyplot as plt


def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.00001)


def OnPolicyTrainingLoop(agent, env, num_steps, view=False):
    frame_idx    = 0
    state = env.reset()
    training_rewards = []
    cum_lengths = []
    ep_reward = 0

    while frame_idx < 50000:
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
                cum_lengths.append(frame_idx)
                ep_reward = 0
                state = env.reset()
                
            if frame_idx % 1000 == 0 and view:
                plot(frame_idx, training_rewards)
            
        agent.train(next_state)

    return cum_lengths, training_rewards


reward_scale = 100

def OffPolicyTrainingLoop(agent, env, training_steps=10000, view=True):
    
    lengths, rewards = [], []
    state, done = env.reset(), False
    ep_score, ep_steps = 0, 0
    for t in range(1, training_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        done = 0 if ep_steps + 1 == 200 else float(done)
        agent.memory.add(state, action, next_state, reward, done)  
        ep_score += reward
        ep_steps += 1
        state = next_state
        
        agent.train()
        
        if done:
            lengths.append(ep_steps)
            rewards.append(ep_score)
            state, done = env.reset(), False
            print("Step: {}, Episode :{}, Score : {:.1f}".format(t, len(lengths), ep_score))
            ep_score, ep_steps = 0, 0
        
        
        if t % 1000 == 0 and view:
            plot(t, rewards)
        
    return lengths, rewards


def observe(env, memory, observation_steps):
    time_steps = 0
    state = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        memory.add(state, action, next_state, reward, done)  

        state = next_state
        time_steps += 1

        if done:
            state = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

    print("")


if __name__ == '__main__':
    main()