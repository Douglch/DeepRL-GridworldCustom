import gym
from stable_baselines3 import PPO

import gridworldcustom
from utils.logger import logger

SIZE = 5
TARGETS = 3

# Change the timestep to the model you want to load
TIMESTEPS = 100_000

# Edit the model_fn to the algorithm you want to use
model_fn = PPO

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=TARGETS)
env.reset()
log = logger(env)

models_dir = f"models/{model_fn.__name__}"
# Edit the number (timestep) to the model you want to load
model_path = f"{models_dir}/{TIMESTEPS}"

model = model_fn.load(model_path, env=env, print_system_info=True)


episodes = 10
total_rewards = []
total_steps = []
distances = {}
for episode in range(1, episodes + 1):
    obs = env.reset()
    terminated = False
    reward_ep = 0
    steps = 0
    distances[f"episode_{episode}"] = []
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = env.step(action)
        distances[f"episode_{episode}"].append(info["shortest distance"])
        reward_ep += reward
        steps += 1
        env.render()
    total_rewards.append(reward_ep)
    total_steps.append(steps)
    print(f"Episode: {episode}, Reward_EP: {reward_ep}, Steps: {steps}")

print("distances:", distances)
print("total_rewards:", total_rewards)
print("total_steps:", total_steps)
env.close()
