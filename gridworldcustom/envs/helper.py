from gym.spaces import Dict, Box


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
