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

Presets
-------

In the ``autonomous-learning-library``, agents are *compositional*, which means that the behavior of a given ``Agent`` depends on 

