import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldCustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=5, agents=1):
        self.size = size  # The size of the square grid (5 * 5)
        self.window_size = 512  # The size of the PyGame window
        self.agents = agents  # The number of agents in the environment
        '''
        The observation is a value representing the agent's current position as
        current_row * nrows + current_col (where both the row and col start at 0).
        For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
        The number of possible observations is dependent on the size of the map.
        '''
        self.state_space = spaces.Discrete(
            size * size)  # The number of possible grids

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_1": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_2": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_3": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location,
                "target_1": self._target_location_1,
                "target_2": self._target_location_2,
                "target_3": self._target_location_3,
                }

    def _get_info(self):
        info = {
            "distancefromtarget 1": np.linalg.norm(
                self._agent_location - self._target_location_1, ord=1
            ),
            "distance from target 2": np.linalg.norm(
                self._agent_location - self._target_location_2, ord=1
            ),
            "distance from target 3": np.linalg.norm(
                self._agent_location - self._target_location_3, ord=1
            ),
        }
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        # Choose the agent's location uniformly at random
        self._agent_location = np.random.randint(
            0, self.size, size=2, dtype=int)

        target_locations = [self._agent_location]

        for i in range(3):  # Check if agent and the 3 rewards are not in the same location
            target_location = np.random.randint(
                0, self.size, size=2, dtype=int)
            while np.any([np.array_equal(target_location, target_locations[j]) for j in range(i+1)]):
                target_location = np.random.randint(
                    0, self.size, size=2, dtype=int)
            target_locations.append(target_location)

        self._target_location_1 = target_locations[1]
        self._target_location_2 = target_locations[2]
        self._target_location_3 = target_locations[3]
        self._targets_visited = np.array([False, False, False])

        observation = self._get_obs()
        # info = self._get_info()
        # print(isinstance(
        #     observation, dict), "<-- observation is a dictionary")
        # print("observation:", observation)
        # if self.render_mode == "human":
        #     self._render_frame()
        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        # self.size is the size of the map
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # # Check if the agent has visited the next target location
        if np.array_equal(self._agent_location, self._target_location_1):
            self._targets_visited[0] = True
        elif np.array_equal(self._agent_location, self._target_location_2):
            self._targets_visited[1] = True
        elif np.array_equal(self._agent_location, self._target_location_3):
            self._targets_visited[2] = True

        # An episode is done iff the agent has reached the target
        terminated = bool(np.all(self._targets_visited))
        # terminated = np.array_equal(self._agent_location, self._target_location_1)

        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, info

    def render(self, mode=None):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target 1
        pygame.draw.rect(
            canvas,
            (155, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location_1,
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 155, 0),
            pygame.Rect(
                pix_square_size * self._target_location_2,
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 0, 155),
            pygame.Rect(
                pix_square_size * self._target_location_3,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
