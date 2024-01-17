import sys
from matplotlib import pyplot as plt
from utils import plot, plot_final


def OnPolicyTrainingLoop_eps(agent, env, batch_eps=1, view=False, algorithm="None"):
    frame_idx    = 0
    training_rewards = []
    cum_lengths = []
    ep_reward = 0

    while frame_idx < 50000:
        for ep in range(batch_eps):
            (state, _), done = env.reset(), False
            while not done:
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                agent.replay_buffer.add(state, action, next_state, reward/100, done)
        
                ep_reward += reward
                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0 and view:
                    plot(frame_idx, training_rewards)
        
            print(f"{frame_idx} -> Episode reward: ", ep_reward)
            training_rewards.append(ep_reward)
            cum_lengths.append(frame_idx)
            ep_reward = 0
    
        agent.train()
    plot_final(frame_idx, training_rewards, algorithm)

    return cum_lengths, training_rewards


def OffPolicyTrainingLoop(agent, env, training_steps=10000, view=True, algorithm=""):
    lengths, rewards = [], []
    (state, _), done = env.reset(), False
    ep_score, ep_steps = 0, 0
    for t in range(1, training_steps):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        done = 0 if ep_steps + 1 == 200 else float(done)
        agent.replay_buffer.add(state, action, next_state, reward, done)  
        ep_score += reward
        ep_steps += 1
        state = next_state
        
        agent.train()
        
        if done:
            lengths.append(ep_steps)
            rewards.append(ep_score)
            (state, _), done = env.reset(), False
            print("Step: {}, Episode :{}, Score : {:.1f}".format(t, len(lengths), ep_score))
            ep_score, ep_steps = 0, 0
        
        
        if t % 1000 == 0 and view:
            plot(t, rewards)

    plot_final(t, rewards, algorithm)
        
    return lengths, rewards



def observe(env, replay_buffer, observation_steps):
    time_steps = 0
    state, _ = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated

        replay_buffer.add(state, action, next_state, reward, done)  

        state = next_state
        time_steps += 1

        if done:
            state, _ = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

    print("")


if __name__ == '__main__':
    main()