

def ContinuousTrainingLoop(agent, env):
    
    score = 0.0
    print_interval = 20
    steps = 0
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        ep_score = 0
        while not done:
            a = agent.act(s)
            s_prime, r, done, info = env.step([a])
            steps += 1
            agent.memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            ep_score += r
            s = s_prime
                
            agent.train()
        
        print("Step: {}, Episode :{}, Score : {:.1f}".format(steps, n_epi, ep_score))
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("Step: {}, # of episode :{}, avg score : {:.1f}".format(steps, n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()