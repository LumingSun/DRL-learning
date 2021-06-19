import gym

env = gym.make('CartPole-v0')
observation = env.reset()
for step in range(100):
    env.render()
    action = env.action_space.sample()
    observation,reward,done,info = env.step(action)
    print(reward)