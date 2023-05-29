from gym.envs.registration import register

register(
    id="gridworldcustom/GridWorldCustom-v0",
    entry_point="gridworldcustom.envs:GridWorldCustomEnv",
)
