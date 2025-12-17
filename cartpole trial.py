import gym
max_ep = 10

for ep_cnt in range(max_ep):
    env = gym.make('CartPole-v1')
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {ep_cnt + 1}: Total Reward: {total_reward}")