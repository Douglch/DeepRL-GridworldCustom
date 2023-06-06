### Deep Reinforcement Learning for GridworldCustom

This is a Deep Reinforcement Learning implementation for GridworldCustom. The implementation is based on the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by DeepMind.

## Installation

The customized environment has to be built from source, you only need to do this once.

```bash
python setup.py build install
```

After installation, you'll see a new folder named `gridworld_custom.egg-info` in the current directory.

## Usage

To see how the model trains, you can follow the instructions on this [site](https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/) to learn how to run the environment on Windows.

Replace the render_mode in `QAgent.py` with `human` to see the model train.

After installation, you can run the model via Tabular method by running

```bash
python QAgent.py
```

You can also train the models with different algorithms in `RLAgent-train.py`

```bash
python RLAgent-train.py
```

and simply change and load the timesteps you want in RLAgent-load.py then run the model to see how the model performs by running

```bash
python RLAgent-load.py
```
