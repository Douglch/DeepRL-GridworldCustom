import numpy as np
import random


class rl_algorithms():

    def __init__(self, env) -> None:
        self.__env__ = env

    def _initialize_q_table(self, state_space, action_space):
        Qtable = np.zeros((state_space, action_space))
        return Qtable

    def greedy_policy(self, Qtable, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(Qtable[state][:])
        return action

    def epsilon_greedy_policy(self, Qtable, state, epsilon):
        # Randomly generate a number between 0 and 1
        random_int = random.uniform(0, 1)
        # if random_int > greater than epsilon --> exploitation
        if random_int > epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = self.greedy_policy(Qtable, state)
        # else --> exploration
        else:
            action = self.env.action_space.sample()

        return action
