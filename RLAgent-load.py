import gym
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

import gridworldcustom
from utils.logger import logger

SIZE = 10
TARGETS = 1
SUCCESS_MODEL = "OneTarget10x10"

# Change the timestep to the model you want to load
TIMESTEPS = SUCCESS_MODEL  # 2_000_000

# Edit the model_fn to the algorithm you want to use
model_fn = PPO

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=TARGETS)
# The Monitor wrapper allows to keep track of the training reward and other infos (useful for plotting)
env.reset()
env = Monitor(env)
# log = logger(env)

models_dir = f"models/{model_fn.__name__}"
# Edit the number (timestep) to the model you want to load
model_path = f"{models_dir}/{TIMESTEPS}"

model = model_fn.load(model_path, env=env, print_system_info=True)

# mean_reward, std_reward = evaluate_policy(
#     model, env, n_eval_episodes=50, deterministic=True)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

episodes = 50
total_rewards = []
total_steps = []
distances = {}
ep_with_mistakes = 0
for episode in tqdm(range(1, episodes + 1)):
    obs = env.reset()
    terminated = False
    mistake = False
    reward_ep = 0
    steps = 0
    distances[f"episode_{episode}"] = []
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = env.step(action)
        # distances[f"episode_{episode}"].append(info["shortest distance"])
        reward_ep += reward
        if reward < 0:
            mistake = True
        print("reward:", reward)
        # print("reward_ep:", reward_ep)
        steps += 1
        env.render()
    if mistake:
        ep_with_mistakes += 1
    total_rewards.append(reward_ep)
    total_steps.append(steps)
    print(f"Episode: {episode}, Reward_EP: {reward_ep}, Steps: {steps}")

# print("distances:", distances)
print("total_rewards:", total_rewards)
print("total_steps:", total_steps)
print(f"episodes with mistakes: {ep_with_mistakes} / {episodes}")
p = 100 * float(ep_with_mistakes) / float(episodes)
print("success rate:", 100 - p, "%")
env.close()
