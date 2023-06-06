import gym
from stable_baselines3 import A2C, PPO

import gridworldcustom
from utils.logger import logger

SIZE = 5
TIMESTEPS = 90000

# Edit the model_fn to the algorithm you want to use
model_fn = PPO

env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=SIZE)
env.reset()
log = logger(env)

models_dir = f"models/{model_fn.__name__}"
# Edit the number (timestep) to the model you want to load
model_path = f"{models_dir}/{TIMESTEPS}"

model = model_fn.load(model_path, env=env, print_system_info=True)


episodes = 1000
for episode in range(1, episodes + 1):
    obs = env.reset()
    terminated = False

    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = env.step(action)
        env.render()

env.close()
