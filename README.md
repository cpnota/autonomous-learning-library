# The Autonomous Learning Library: An Object-Oriented Deep Reinforcement Learning Library in Pytorch

The Autonomous Learning Library (`all`) is an object-oriented deep reinforcement learning library in `pytorch`. The goal of the library is to provide implementations of modern reinforcement learning algorithms that reflect the way that reinforcement learning researchers think about agent design and to provide the components necessary to build and test new ideas with minimal overhead.

## Why use `all`?

The primary reason for using `all` over its many competitors is because it contains components that allow you to *build your own* reinforcement learning agents.
We provide out-of-the-box modules for:

- [x] Custom Q-Networks, V-Networks, policy networks, and feature networks
- [x] Generic function approximation
- [x] Target networks
- [x] Polyak averaging
- [x] Experience Replay
- [x] Prioritized Experience Replay
- [x] Advantage Estimation
- [x] Generalized Advantage Estimation (GAE)
- [x] Easy parameter and learning rate scheduling
- [x] An enhanced `nn` module (includes dueling layers, noisy layers, action bounds, and the coveted `nn.Flatten`)
- [x] `gym` to `pytorch` wrappers
- [x] Atari wrappers
- [x] An `Experiment` API for comparing and evaluating agents
- [x] A `SlurmExperiment` API for running massive experiments on computing clusters
- [x] A `Writer` object for easily logging information in `tensorboard`
- [x] Plotting utilities for generating paper-worthy result plots

Rather than being embedded in the agents, all of these modules are available for use by your own custom agents.
Additionally, the included agents accept custom versions of any of the above objects.
Have a new type of replay buffer in mind?
Code it up and pass it directly to our `DQN` and `DDPG` implementations.
Additionally, our agents were written with readibility as a primary concern, so they are easy to modify.

## Algorithms

As of today, `all` contains implementations of the following deep RL algorithms:

- [x] Advantage Actor-Critic (A2C)
- [x] Categorical DQN (C51)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [x] Deep Q-Learning (DQN) + extensions
- [x] Proximal Policy Optimization (PPO)
- [x] Rainbow (Rainbow)
- [x] Soft Actor-Critic (SAC)

It also contains implementations of the following "vanilla" agents, which provide useful baselines and perform better than you may expect:

- [x] Vanilla Actor-Critic
- [x] Vanilla Policy Gradient
- [x] Vanilla Q-Learning
- [x] Vanilla Sarsa

We will try to stay up-to-date with advances in the field, but we do not intend to implement every algorithm. Rather, we prefer to maintain a smaller set of high-quality agents that have achieved notoriety in the field.

We have labored to make sure that our implementations produce results comparable to published results.
Here's a sampling of performance on several Atari games:

![atari40](atari40.png)

These results were generated using the `all.presets.atari` module, the `SlurmExperiment` utility, and the `all.experiments.plots` module.

## Example

Our agents implement a single method: `action = agent.act(state, reward)`.
Much of the complexity is handled behind the scenes, making the final API simple.
Unlike many libraries, we do not combine the learning algorithm and the training loop.
Instead, our agents can be embedded in existing applications and trained in the online setting.

The `all.presets` includes agents that preconfigured for certain types of environments.
It can be used as follows:

```python
from all.presets.atari import dqn
from all.environments import AtariEnvironment

env = AtariEnvironment('Breakout')
agent = dqn(lr=3e-4)(env)

while True:
    if env.done:
        env.reset()
    else:
        env.step(action)
    env.render()
    action = agent.act(env.state, env.reward)
```

However, generally we recommend using the `Experiment` API, which adds many additional features:

```python
from all.presets.atari import a2c, dqn
from all.environments import AtariEnvironment
from all.experiments import Experiment

# use graphics card for much faster training
device = 'cuda'
experiment = Experiment(AtariEnvironment('Breakout', device=device), frames=10e6)
experiment.run(a2c(device=device))
experiment.run(dqn(device=device))
```

Results can be viewed by typing:

```
make tensorboard
```

## Installation

This library is built on top of `pytorch`.
If you don't want your trials to take forever, it is highly recommended that you make sure your installation has CUDA support (and that you have a CUDA compatible GPU).
You'll also need `tensorflow` in order to use `tensorboard` (used for storing and plotting runs).

There are two ways to install the `autonomous-learning-library` : a "light" installation, which assumes that the major dependencies are already installed, and a "full" installation which installs everything from scratch.

### Light Installation

Use this if you already have `pytorch` and `tensorflow` installed.
Simply run:

```bash
pip install -e .
```

Presto! If you have any trouble with installing the Gym environments, check out their [GitHub page](https://github.com/openai/gym) and try whatever they recommend in [current year].

### Full Installation

If you're on Linux and don't have `pytorch` or `tensorflow` installed, we did you the courtesy of providing a helpful install script:

```bash
make install
```

With any luck, the `all` library should now be installed!

### Testing Your Installation

The unit tests may be run using:

```
make test
```

If the unit tests pass with no errors, it is more than likely that your installation works! The unit test run every agent using both `cpu` and `cuda` for a few timesteps/episodes.

## Running the Presets

You can easily benchmark the included algorithms using the scripts in `./benchmarks`.
To run a simple `CartPole` benchmark, run:

```
python scripts/classic.py CartPole-v1 dqn
```

Results are printed to the console, and can also be viewed by running:

```
make tensorboard
```

and opening your browser to http://localhost:6006.

To run an Atari benchmark in CUDA mode (warning: this could take several hours to run, depending on your machine):

```
python scripts/atari.py Pong dqn
```

If you want to run in `cpu` mode (~10x slower on my machine), you can add ```--device cpu```:

```
python scipts/atari.py Pong dqn --device cpu
```

## Note

This library was built at the [Autonomous Learning Laboratory](http://all.cs.umass.edu) (ALL) at the [University of Massachusetts, Amherst](https://www.umass.edu).
It was written and is currently maintained by Chris Nota (@cpnota).
The views expressed or implied in this repository do not necessarily reflect the views of the ALL.
