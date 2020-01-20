Building Your Own Agent
=======================

In the previous section, we discussed the basic components of the ``autonomous-learning-library``.
While the library contains a selection of preset agents, the primary goal of the library is to be a tool to build *your own* agents.
To this end, we have provided an `example project <https://github.com/cpnota/all-example-project>`_ containing a new *model predictive control* variant of DQN to demonstrate the flexibility of the library.
Briefly, when creating your own agent, you will generally have the following components:

1. An ``agent.py`` file containing the high-level implementation of the ``Agent``.
2. A ``model.py`` file containing the PyTorch models appropriate for your chosen domain.
3. A ``preset.py`` file that composes your ``Agent`` using the appropriate model and other objects.
4. A ``main.py`` or similar file that runs your agent and any ``autonomous-learning-library`` presets you wish to compare against.

While it is not necessary to follow this structure, we believe it will generally guide you towards using the ``autonomous-learning-library`` in the intended manner and ensure that your code is understandable to other users of the library.
