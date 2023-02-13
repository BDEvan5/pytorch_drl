import sys

reward_scale = 100

def ContinuousTrainingLoop(agent, env):
    
    steps = 0
    lengths, rewards = [], []
    for n_epi in range(20):
        state = env.reset()
        done = False
        
        ep_score = 0
        ep_steps = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            done = 0 if ep_steps + 1 == 200 else float(done)
            agent.memory.add(state, action, next_state, reward, done)  
            
            ep_score += reward
            ep_steps += 1
            state = next_state
                
            agent.train()
        
        lengths.append(steps)
        rewards.append(ep_score)
        
        print("Step: {}, Episode :{}, Score : {:.1f}".format(steps, n_epi, ep_score))
        
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