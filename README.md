# The Autonomous Learning Library

This is a library for reinforcement learning built at the [Autonomous Learning Laboratory](http://all.cs.umass.edu) (ALL) at the [University of Massachusetts, Amherst](https://www.umass.edu).
The goal of the library is to provide implementations of modern reinforcement learning algorithms that reflect the way that reinforcement learning researchers think about agent design, and to provide the components necessary to build and test new types of agents.

This library was written and is currently maintained by Chris Nota (@cpnota).
The views expressed or implied in this repository do not necessarily reflect the views of the ALL.

## Installation

This library is built on top of `pytorch`.
If you don't want your trials to take forever, it is highly recommended that you make sure your installation has CUDA support (and that you have a CUDA compatible GPU).
You'll also need `tensorflow` in order to use `tensorboard` (used for storing and plotting runs).

There are two ways to install the `autonomous-learning-library`: a "light" installation, which assumes that the major dependencies are already installed, and a "full" installation which installs everything from scratch.

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

## Running the Benchmarks

You can easily benchmark the included algorithms using the scripts in `./benchmarks`.
To run a simple `CartPole` benchmark, run:

```
python benchmarks/classic.py CartPole-v1 dqn
```

Results are printed to the console, and can also be viewed by running:

```
make tensorboard
```

and opening your browser to http://localhost:6006.

To run an Atari benchmark in CUDA mode (warning: this could take several hours to run, depending on your machine):

```
python benchmarks/atari.py Pong dqn
```

If you want to run in `cpu` mode (~10x slower on my machine), you can add ```--device cpu```:

```
python benchmarks/atari.py Pong dqn --device cpu
```

Finally, to run the entire benchmarking suite:

```
python benchmark/release.py
```

Before every merge to master, we re-run the benchmarking suite and commit the results to this repository.
The results are labeled with the has of the commit where the benchmark command was run.
Your can view our results by running `make benchmark`, and opening your browser to http://localhost:6007. 
To replicate these results, you can checkout that specific commit that the run is labeled with and run the above command.
