import gym
from stable_baselines3 import PPO

import os
import gridworldcustom
from utils.logger import logger

# Training parameters
TIMESTEPS = 200_000
SIZE = 10
TARGETS = 5

model_fn = PPO
models_dir = f"models/{model_fn.__name__}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE, targets=TARGETS)
env.reset()
# log = logger(env)
# log.print_obs_action_space()

model = model_fn("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

for i in range(1, 11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                tb_log_name=f"{model_fn.__name__}")
    # Save the model every 100 timesteps
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
