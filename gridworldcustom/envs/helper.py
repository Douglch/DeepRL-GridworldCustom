from gym.spaces import Dict, Box
from random import randrange
import colorsys
import random


class Helper():
    def __init__(self) -> None:
        pass

    def create_obs(obs_size, agents):
        obs = Dict(
            {
                "agent": Box(0, obs_size - 1, shape=(2,), dtype=int)
            }
        )
        for i in range(agents):
            obs[f"target_{i + 1}"] = Box(0,
                                         obs_size - 1, shape=(2,), dtype=int)

        return obs

    def predefined_rgb():
        # Number of colors to generate
        num_colors = 100

        # Define the range of hue, saturation, and value for randomization
        hue_range = (0, 1)  # Range of hue (0 to 1)
        saturation_range = (0, 1)  # Range of saturation (0 to 1)
        value_range = (0, 0.5)  # Range of value (0 to 0.5 for darker colors)

        # Precompute the random dark colors
        dark_colors = []
        for _ in range(num_colors):
            # Generate random values for hue, saturation, and value
            hue = random.uniform(*hue_range)
            saturation = random.uniform(*saturation_range)
            value = random.uniform(*value_range)

            # Convert HSV to RGB and scale to 0-255
            rgb = tuple(int(val * 255)
                        for val in colorsys.hsv_to_rgb(hue, saturation, value))

            # Add the RGB tuple to the list
            dark_colors.append(rgb)

        return dark_colors
