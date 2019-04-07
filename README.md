# The Autonomous Learning Library

This is a library for reinforcement learning built at the [Autonomous Learning Laboratory](http://all.cs.umass.edu) (ALL) at the [University of Massachusetts, Amherst](https://www.umass.edu).
The goal of the library is to provide implementations of modern reinforcement learning algorithms that reflect the way that reinforcement learning researchers think about agent design, and to provide the components necessary to build and test new types of agents.

This library was written and is currently maintained by Chris Nota (@cpnota).
The views expressed or implied in this repository do not necessarily reflect the views of the ALL.

## Installation

First, clone the repository:

```
git clone https://github.com/cpnota/autonomous-learning-library.git
cd autonomous-learning-library
```

Next, you must install all of the dependencies.
Python's dependency management tools are really bad, and CUDA makes everything worse.
There's not much we can do about this.
The following instructions should work on most machines, and were tested on Ubuntu 18.04:

1. For GPU support, install [CUDA](https://docs.nvidia.com/cuda/index.html) 10.1 on your machine. Other CUDA versions may work, but I have not tested them. If you haven't installed CUDA before, this step is likely the trickiest.
2. Install [Anaconda](https://www.anaconda.com) for Python 3.7.
4. Install pytorch. If you've followed along so far: `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`.
5. Install tensorflow. This is necessary for running `tensorboard`, which is used for displaying results: `conda install tensorflow`.
6. Finally, install the rest of the dependencies and the module itself: `make install`
7. (Sometimes) This `gym[atari]` dependency doesn't always seem to install property. This can be fixed by running `pip install gym[atari]` after installing thest rest of the repo.

To test your installation, run:

```
make test
```

If the unit tests pass with no errors, you're good to go!

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
python benchmarks/atari.py Pong dqn cuda
```

Finally, to run the entire benchmarking suite:

```
python benchmark/release.py
```

Before every merge to master, we re-run the benchmarking suite and commit the results to this repository.
The results are labeled with the has of the commit where the benchmark command was run.
Your can view our results by running `make benchmark`, and opening your browser to http://localhost:6007. 
To replicate these results, you can checkout that specific commit that the run is labeled with and run the above command.
