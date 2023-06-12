import gym
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
import time

import os
import gridworldcustom
from utils.logger import logger

# Training parameters
TIMESTEPS = 200_000
SIZE = 10
TARGETS = 4
EPISODES = 10

model_fn = PPO
models_dir = f"models/{model_fn.__name__}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

start_time_vec_env = time.time()

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=TARGETS)
env.reset()
# print(env)
# # log = logger(env)
# # log.print_obs_action_space()

model = model_fn("MultiInputPolicy", env,
                 verbose=1, tensorboard_log=log_dir)

for i in range(1, EPISODES + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                tb_log_name=f"{model_fn.__name__}")
    # Save the model every TIMESTEPS timesteps
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()

time_vec_env = time.time() - start_time_vec_env
print(f"Took {time_vec_env:.2f}s")
