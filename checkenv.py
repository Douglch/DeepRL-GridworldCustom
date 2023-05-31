import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

import gridworldcustom

SIZE = 5

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE)
# It will check your custom environment and output additional warnings if needed
check_env(env)

episodes = 10

# for episodes in range(episodes):
#     terminated = False
#     obs, info = env.reset()
#     while not terminated:
#         random_action = env.action_space.sample()
#         print("action:", random_action)
#         obs, reward, terminated, info = env.step(random_action)
#         print("reward:", reward)

# reset_returns = env.reset()
# obs, info = reset_returns
# print(isinstance(obs, dict))

env.close()
