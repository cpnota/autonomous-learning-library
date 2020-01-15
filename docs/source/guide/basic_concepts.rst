Basic Concepts
==============

In this section, we explain the basic elements of the ``autonomous-learning-library`` and the philosophy behind some of the basic design decision.

Agent-Based Design
------------------


One of the core philosophies in the autonomous-learning-library is that RL should be agent-based, not algorithm-based.
To see what we mean by this, check out the OpenAI Baselines implementation of DQN.
There's a giant function called learn which accepts an environment and a bunch of hyperparameters at the heart of which there is a control loop which calls many different functions.
Which part of this function is the agent? Which part is the environment? Which part is something else?
We call this implementation algorithm-based because the central abstraction is a function called learn which provides the complete specification if an algorithm.
What should the proper abstraction for agent be, then? We have to look no further than the following famous diagram:

.. image:: ./rl.jpg

The definition of an ``Agent`` is simple.
It accepts a state and a reward and returns an action.
That's it.
Everything else is an implementation detail.
Here's the ``Agent`` interface in the autonomous-learning-library:

.. code-block:: python

    class Agent(ABC):
        @abstractmethod
        def act(self, state, reward):
            pass

That's it.
When and how it trains is nobody's business except the ``Agent`` itself.
When the ``Agent`` is allowed to act is determined by the control loop.
What might an implementation of this look like? Here's the act function from our DQN implementation:

.. code-block:: python

    def act(self, state, reward):
        self._store_transition(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self.policy(state)
        return self.action

That's it. ``_store_transition()`` and ``_train()`` are private helper methods.
There is no reason for the control loop to know anything about these details.
This approach simplifies both our ``Agent`` implementation and the control loop itself.

Separating the control loop logic from the ``Agent`` logic allows greater flexibility in the way agents are used.
In fact, ``Agent`` is entirely decoupled from the ``Environment`` interface.
This means that our agents can be used outside of standard research environments, such as by being part of a REST api, a multi-agent system, etc.
Any code that passes a ``State`` and a reward is compatible with our agents.

Function Approximation
----------------------

Almost everything a deep reinforcement learning agent does is predicated on *function approximation*.

.. image:: ./approximation.jpeg

For this reason, one of the central abstractions in the ``autonomous-learning-library`` is ``Approximation``.
By building agents that rely on the ``Approximation`` abstraction rather than directly interfacing with PyTorch ``Module`` and ``Optimizer`` objects,
we can add to or modify the functionality of an Agent without altering its source code (this is known as the `Open-Closed Principle <https://en.wikipedia.org/wiki/Open–closed_principle>`_).
The default ``Approximation`` object allows us to achieve a high level of code reuse by encapsulating common functionality such as logging, model checkpointing, target networks, learning rate schedules and gradient clipping. The Approximation object in turn relies on a set of abstractions that allow users to alter its behavior.
Let's look at a simple usage of ``Approximation`` in solving a very easy supervised learning task:

.. code-block:: python

    import torch
    from torch import nn, optim
    from all.approximation import Approximation

    # create a pytorch module
    model = nn.Linear(16, 1)

    # create an associated pytorch optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # create the function approximator
    f = Approximation(model, optimizer)

    for _ in range(200):
        # Generate some arbitrary data.
        # We'll approximate a very simple function:
        # the sum of the input features.
        x = torch.randn((16, 16))
        y = x.sum(1, keepdim=True)

        # forward pass
        y_hat = f(x)

        # compute loss
        loss = nn.functional.mse_loss(y_hat, y)

        # backward pass
        f.reinforce(loss)

Easy! Now let's look at the _train() function for our DQN agent:

.. code-block:: python

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

            # forward pass
            values = self.q(states, actions)
            targets = rewards + self.discount_factor * torch.max(self.q.target(next_states), dim=1)[0]

            # compute loss
            loss = mse_loss(values, targets)

            # backward pass
            self.q.reinforce(loss)

Just as easy!
The agent does not need to know anything about the network architecture, logging, regularization, etc.
These are all handled through the appropriate configuration of ``Approximation``.
Instead, the ``Agent`` implementation is able to focus exclusively on its sole purpose: defining the RL algorithm itself.
By encapsulating these details in ``Approximation``, we are able to follow the `single responsibility principle <https://en.wikipedia.org/wiki/Single_responsibility_principle>`_.

A few other quick things to note: ``f.eval(x)`` runs a forward pass in ``torch.no_grad()``.
``f.target(x)`` calls the *target network* (an advanced concept used in algorithms such as DQN. S, for example, David Silver's `course notes <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Talks_files/deep_rl.pdf>`_) associated with the ``Approximation``, also with ``torch.no_grad()``.
The ``autonomous-learning-library`` provides a few thin wrappers over ``Approximation`` for particular purposes, such as ``QNetwork``, ``VNetwork``, ``FeatureNetwork``, and several ``Policy`` implementations.

Environment
-----------

The importance of the ``Environment`` in reinforcement learning nearly goes without saying.
In the ``autonomous-learning-library``, the prepackaged environments are simply wrappers for `OpenAI Gym <http://gym.openai.com>`_, the defacto standard library for RL environments.

.. figure:: ./ale.png

    Some environments included in the Atari suite in Gym. This picture is just so you don't get bored.


We add a few additional features:

1) ``gym`` primarily uses ``numpy.array`` for representing states and actions. We automatically convert to and from ``torch.Tensor`` objects so that agent implemenetations need not consider the difference.
2) We add properties to the environment for ``state``, ``reward``, etc. This simplifies the control loop and is generally useful.
3) We apply common preprocessors, such as several standard Atari wrappers. However, where possible, we prefer to perform preprocessing using ``Body`` objects to maximize the flexibility of the agents.

Below, we show how several different types of environments can be created:

.. code-block:: python

    from all.environments import AtariEnvironment, GymEnvironment

    # create an Atari environment on the gpu
    env = AtariEnvironment('Breakout', device='cuda')

    # create a classic control environment on the compute
    env = GymEnvironment('CartPole-v0')

    # create a PyBullet environment on the cpu
    import pybullet_envs
    env = GymEnvironment('HalfCheetahBulletEnv-v0')

Now we can write our first control loop:

.. code-block:: python

    # initialize the environment
    env.reset()

    # Loop for some arbitrary number of timesteps.
    for timesteps in range(1000000):
        env.render()
        action = agent.act(env.state, env.reward)
        env.step(action)

        if env.done:
            # terminal update
            agent.act(env.state, env.reward)

            # reset the environment
            env.reset()

Of course, this control loop is not exactly feature-packed.
Generally, it's better to use the ``Experiment`` API described later.


Presets
-------

In the ``autonomous-learning-library``, agents are *compositional*, which means that the behavior of a given ``Agent`` depends on 

