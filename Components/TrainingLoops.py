import sys

reward_scale = 100

def ContinuousTrainingLoop(agent, env):
    
    score = 0.0
    print_interval = 20
    steps = 0
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        ep_score = 0
        ep_steps = 0
        while not done:
            a = agent.act(s)
            s_prime, r, done, info = env.step(a)
            steps += 1
            
            done = 0 if ep_steps + 1 == 200 else float(done)
            agent.memory.put((s, a, s_prime, r, done))
            
            score +=r
            ep_score += r
            ep_steps += 1
            s = s_prime
                
            agent.train()
        
        print("Step: {}, Episode :{}, Score : {:.1f}".format(steps, n_epi, ep_score))
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("Step: {}, # of episode :{}, avg score : {:.1f}".format(steps, n_epi, score/print_interval))
            score = 0.0

    env.close()


def observe(env,replay_buffer, observation_steps):
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.put((obs, action, new_obs, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


if __name__ == '__main__':
    main()