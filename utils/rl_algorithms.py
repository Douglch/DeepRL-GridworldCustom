import numpy as np
import random


class RLAlgorithms():

    def __init__(self, env, size) -> None:
        self.__env__ = env
        self.__size__ = size

    def _initialize_q_table(self, action_space):
        Qtable = {}
        for x in range(0, self.__size__):
            for y in range(0, self.__size__):
                Qtable[(x, y)] = [np.random.uniform(0, 0)
                                  for i in range(action_space.n)]
        return Qtable

    def greedy_policy(self, Qtable, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(Qtable[state])
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
            action = self.__env__.action_space.sample()

        return action

    # def update_q_table(self, Qtable, state, action, reward, new_state, lr, gamma):
    #     new_q_value = np.max(Qtable[new_state])
    #     Qtable[state][action] += lr * \
    #         (reward + gamma *
    #          new_q_value - Qtable[state][action])
    #     return Qtable
