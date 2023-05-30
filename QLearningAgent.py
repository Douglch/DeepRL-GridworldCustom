import gridworldcustom

import gym
import numpy as np
from tqdm import tqdm
from gym.wrappers import TimeLimit

MAX_STEPS = 99
SHOW_EVERY = 5000

# Training parameters
N_TRAINING_EPISODES = 25000   # Total training episodes
LEARNING_RATE = 0.7           # Learning rate

# Evaluation parameters
N_EVAL_EPISODES = 100        # Total number of test episodes
NUM_EPISODES = 100
MAX_STEPS = 99               # Max steps per episode
GAMMA = 0.95                 # Discounting rate

# Exploration parameters
MAX_EPSILON = 1.0             # Exploration probability at start
MIN_EPSILON = 0.05           # Minimum exploration probability
DECAY_RATE = 0.005            # Exponential decay rate for exploration prob


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.state_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        max_q_value = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * \
            (reward + self.gamma * max_q_value - self.q_table[state, action])

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            state, info = self.env.reset()
            state = state['agent']
            terminated = False
            total_reward = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, info = self.env.step(
                    action)
                total_reward += reward
                next_state = next_state['agent']
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            if (episode + 1) % 500 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
            if (episode + 1) % SHOW_EVERY == 0:
                self.env.render()

    def test(self, num_episodes=100):
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)

        average_reward = np.mean(total_rewards)
        print(f"Average Reward: {average_reward}")


# Create the environment
env = gym.make("gridworldcustom/GridWorldCustom-v0",
               render_mode="rgb_array", size=5)
# env = TimeLimit(env, max_episode_steps=MAX_STEPS)
env.reset()

# Create the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
agent.train(num_episodes=N_TRAINING_EPISODES)

# Test the agent
agent.test(num_episodes=100)
