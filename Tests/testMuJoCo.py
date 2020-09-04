import gym


# env = gym.make('FetchPush-v1')
# env.reset()
# for _ in range(1000):
# #   env.render()
#     state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
#     print(f"State: {state}")
# print("end")

env = gym.make('HalfCheetah-v2')

obs_shape = env.observation_space.shape[0]
print(obs_shape)
action = env.action_space.shape[0]
print(action)
act_shape = env.action_space
print(act_shape)














