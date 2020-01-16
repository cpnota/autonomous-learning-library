Benchmark Performance
=====================

Reinforcement learning algorithms are difficult to debug and test.
For this reason, in order to ensuring the correctness of the preset agents provided by the ``autonomous-learning-library``,
we benchmark each algorithm after every major change.
We also discuss the performance of our implementations relative to published results.

Atari Presets
-------------

To benchmark the Atari presets, we ran each agent for 10 million timesteps (40 million in-game frames).
The learning rate is decayed over the course of training using cosine annealing.
The environment implementation uses the following wrappers:

* NoopResetEnv (adds a random number of noops at the beginning of each game reset)
* MaxAndSkipEnv (Repeats each action four times before the next agent observation. Takes the max pixel value over the four frames.)
* FireResetEnv (Automatically chooses the "FIRE" action when env.reset() is called)
* WarpFrame (Rescales the frame to 84x84 and greyscales the image)
* LifeLostEnv (Adds a key to "info" indicating that a life was lost)

Additionally, we use the following agent "bodies":

* FrameStack (provides the last four frames as the state)
* ClipRewards (Converts all rewards to {-1, 0, 1})
* EpisodicLives (If life was lost, treats the frame as the end of an episode)

The results were as follows:

.. image:: ../../../benchmarks/atari40.png

For comparison, we look at the results published in the paper, `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_:

.. image:: ./rainbow.png

In these results, the authors ran each agent for 50 million timesteps (200 million frames).
We can see that at the 10 million timestep mark, our results are similar or slightly better.
Our ``dqn`` and ``ddqn`` in particular were better almost across the board.
While there are almost certainly some minor implementation differences,
our agents achieved very similar behavior to the agents tested by DeepMind.
