import gym
from gym import spaces
# from stable_baselines3.common.env_checker import check_env
from gym.utils.env_checker import check_env

import gridworldcustom

SIZE = 15

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=5)
# It will check your custom environment and output additional warnings if needed
env.reset()
check_env(env)

episodes = 10

for episodes in range(episodes):
    terminated = False
    obs = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        print("action:", random_action)
        obs, reward, terminated, info = env.step(random_action)
        print("reward:", reward)
        env.render()

# reset_returns = env.reset()
# obs, info = reset_returns
# print(isinstance(obs, dict))

env.close()
