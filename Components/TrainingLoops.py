import sys

reward_scale = 100

def ContinuousTrainingLoop(agent, env):
    
    score = 0.0
    print_interval = 20
    steps = 0
    for n_epi in range(10000):
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
            # agent.memory.put((s, a, s_prime, r, done))
            
            score += reward
            ep_score += reward
            ep_steps += 1
            state = next_state
                
            agent.train()
        
        print("Step: {}, Episode :{}, Score : {:.1f}".format(steps, n_epi, ep_score))
        
        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("Step: {}, # of episode :{}, avg score : {:.1f}".format(steps, n_epi, score/print_interval))
        #     score = 0.0

    env.close()


def observe(env, memory, observation_steps):
    time_steps = 0
    state = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        memory.add(state, action, next_state, reward, done)  
        # replay_buffer.put((obs, action, new_obs, reward, done))

        state = next_state
        time_steps += 1

        if done:
            state = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


if __name__ == '__main__':
    main()