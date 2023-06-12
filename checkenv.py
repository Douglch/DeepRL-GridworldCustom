import gym
from gym import spaces
# from stable_baselines3.common.env_checker import check_env
from gym.utils.env_checker import check_env
from tqdm import tqdm

import gridworldcustom

SIZE = 5
TARGETS = 2

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=TARGETS)
# It will check your custom environment and output additional warnings if needed
env.reset()
check_env(env)

episodes = 10
total_rewards = []
total_steps = []
distances_step = {}

for episode in range(1, episodes + 1):
    terminated = False
    obs = env.reset()
    print("obs:", obs)
    reward_ep = 0
    steps = 0
    distances_step[f"episode_{episode}"] = []
    while not terminated:
        random_action = env.action_space.sample()
        obs, reward, terminated, info = env.step(random_action)
        # distances_step[f"episode_{episode}"].append(
        #     info["current distance to nearest location"])
        print("current distance to nearest location:",
              info["current distance to nearest location"])
        print("reward:", reward)
        # print("entity location:", info["entity locations"])
        # print("agent location:", info["agent location"])
        reward_ep += reward
        steps += 1
        env.render()
    # total_rewards.append(reward_ep)
    # total_steps.append(steps)
    print(f"Episode: {episode}, Reward_EP: {reward_ep}, Steps: {steps}")

# print("distances:", distances_step)
# print("total_rewards:", total_rewards)
# print("total_steps:", total_steps)
# reset_returns = env.reset()
# obs, info = reset_returns
# print(isinstance(obs, dict))

env.close()
