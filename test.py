import gridworldcustom
from utils.rl_algorithms import rl_algorithms
from utils.logger import logger

import gym
import numpy as np
from tqdm import tqdm


env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="human", size=10)
# num_episodes = 100
# observation, info = env.reset()

# state_space = env.count_states
# print("There are ", state_space, " possible states")

# action_space = env.action_space.n
# print("There are ", action_space, " possible actions")

# Qtable_taxi = _initialize_q_table(state_space, action_space)
# # print(Qtable_taxi)
# print("Q-table shape: ", Qtable_taxi.shape)

# # Training parameters
# n_training_episodes = 25000   # Total training episodes
# learning_rate = 0.7           # Learning rate

# # Evaluation parameters
# n_eval_episodes = 100        # Total number of test episodes
# max_steps = 99               # Max steps per episode
# gamma = 0.95                 # Discounting rate

# # Exploration parameters
# max_epsilon = 1.0             # Exploration probability at start
# min_epsilon = 0.05           # Minimum exploration probability
# decay_rate = 0.005            # Exponential decay rate for exploration prob

# # DO NOT MODIFY EVAL_SEED
# eval_seed = [16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55,
#              161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38,
#              112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
# # Each seed has a specific starting state


# def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
#     for episode in tqdm(range(n_training_episodes)):
#         # Reduce epsilon (because we need less and less exploration)
#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
#             np.exp(-decay_rate*episode)
#         # Reset the environment
#         state, info = env.reset()
#         step = 0
#         terminated = False
#         truncated = False
#         # repeat
#         for step in range(max_steps):
#             # Choose the action At using epsilon greedy policy
#             action = epsilon_greedy_policy(Qtable, state, epsilon)

#             # Take action At and observe Rt+1 and St+1
#             # Take the action (a) and observe the outcome state(s') and reward (r)
#             new_state, reward, terminated, truncated, info = env.step(action)
#             # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
#             print("state: ", state, " action: ", action, " reward: ", reward)
#             Qtable[state][action] = Qtable[state][action] + learning_rate * \
#                 (reward + gamma *
#                  np.max(Qtable[new_state]) - Qtable[state][action])

#             # If terminated or truncated finish the episode
#             if terminated or truncated:
#                 break

#             # Our next state is the new state
#             state = new_state
#     return Qtable


# Qtable_taxi = train(n_training_episodes, min_epsilon,
#                     max_epsilon, decay_rate, env, max_steps, Qtable_taxi)
# Qtable_taxi


# def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
#     """
#     Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
#     :param env: The evaluation environment
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param Q: The Q-table
#     :param seed: The evaluation seed array (for taxi-v3)
#     """
#     episode_rewards = []
#     for episode in tqdm(range(n_eval_episodes)):
#         if seed:
#             state, info = env.reset(seed=seed[episode])
#         else:
#             state, info = env.reset()
#         step = 0
#         truncated = False
#         terminated = False
#         total_rewards_ep = 0

#         for step in range(max_steps):
#             # Take the action (index) that have the maximum expected future reward given that state
#             action = greedy_policy(Q, state)
#             new_state, reward, terminated, truncated, info = env.step(action)
#             total_rewards_ep += reward

#             if terminated or truncated:
#                 break
#             state = new_state
#         episode_rewards.append(total_rewards_ep)
#     mean_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)

#     return mean_reward, std_reward


# mean_reward, std_reward = evaluate_agent(
#     env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
# print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
