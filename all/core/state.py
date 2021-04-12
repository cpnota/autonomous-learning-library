import numpy as np
import torch
import warnings


class State(dict):
    """
    An environment State.

    An environment State represents all of the information available to an agent at a given timestep,
    including the observation, reward, and the done flag.
    The State object contains useful utilities for creating StateArray objects,
    constructing State objects for OpenAI gym environments,
    masking the output of networks based on the done flag, etc.

    Args:
        x (dict):
            A dictionary containing all state information.
            Any key/value can be included, but the following keys are standard:

            observation (torch.tensor) (required):
                A tensor representing the current observation available to the agent

            reward (float) (optional):
                The reward for the previous state/action. Defaults to 0.

            done (bool) (optional):
                Whether or not this is a terminal state. Defaults to False.

            mask (float) (optional):
                The mask (0 or 1) for the current state.
        device (string):
            The torch device on which component tensors are stored.
    """

    def __init__(self, x, device='cpu', **kwargs):
        if not isinstance(x, dict):
            x = {'observation': x}
        for k, v in kwargs.items():
            x[k] = v
        if 'observation' not in x:
            raise Exception('State must contain an observation')
        if 'reward' not in x:
            x['reward'] = 0.
        if 'done' not in x:
            x['done'] = False
        if 'mask' not in x:
            x['mask'] = 1. - x['done']
        super().__init__(x)
        self._shape = ()
        self.device = device

    @classmethod
    def array(cls, list_of_states):
        """
        Construct a StateArray from a list of State or StateArray objects.
        The shape of the resulting StateArray is (N, ...M), where N is the length of the input list
        and M is the shape of the component State or StateArray objects.

        Args:
            list_of_states: A list of State or StateArray objects with a matching shape.

        Returns:
            A StateArray object.
        """
        device = list_of_states[0].device
        shape = (len(list_of_states), *list_of_states[0].shape)
        x = {}

        for key in list_of_states[0].keys():
            v = list_of_states[0][key]
            try:
                if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                    x[key] = torch.stack([torch.stack(state[key]) for state in list_of_states])
                elif torch.is_tensor(v):
                    x[key] = torch.stack([state[key] for state in list_of_states])
                else:
                    x[key] = torch.tensor([state[key] for state in list_of_states], device=device)
            except KeyError:
                warnings.warn('KeyError while creating StateArray for key "{}", omitting.'.format(key))
            except ValueError:
                warnings.warn('ValueError while creating StateArray for key "{}", omitting.'.format(key))
            except TypeError:
                warnings.warn('TypeError while creating StateArray for key "{}", omitting.'.format(key))

        return StateArray(x, shape, device=device)

    def apply(self, model, *keys):
        """
        Apply a model to the state.
        Automatically selects the correct keys, reshapes the input/output as necessary and applies the mask.

        Args:
            model (torch.nn.Module): A torch Module which accepts the components corresponding
                                     to the given keys as args.
            keys (string): Strings corresponding to the desired components of the state.
                           E.g., apply(model, 'observation', 'reward') would pass the observation
                           and reward as arguments to the model.

        Returns:
            The output of the model.
        """
        return self.apply_mask(self.as_output(model(*[self.as_input(key) for key in keys])))

    def as_input(self, key):
        """
        Gets the value for a given key and reshapes it to a batch-style tensor suitable as input to a pytorch module.

        Args:
            key (string): The component of the state to select.

        Returns:
            A torch.tensor containing the value of the component with a batch dimension added.
        """
        return self[key].unsqueeze(0)

    def as_output(self, tensor):
        """
        Reshapes the output of a batch-style pytorch module to match the original shape of the state.

        Args:
            tensor (torch.tensor): The output of a batch-style pytorch module.

        Returns:
            A torch.tensor containing the output in the appropriate shape.
        """
        return tensor.squeeze(0)

    def apply_mask(self, tensor):
        """
        Applies the mask to the given tensor, generally to prevent backpropagation through terminal states.

        Args:
            tensor (torch.tensor): The tensor to apply the mask to.

        Returns:
            A torch.tensor with the mask applied.
        """
        return tensor * self.mask

    def update(self, key, value):
        """
        Adds a key/value pair to the state, or updates an existing key/value pair.
        Note that this is NOT an in-place operation, but returns a new State or StateArray.

        Args:
            key (string): The name of the state component to update.
            value (any): The value of the new state component.

        Returns:
            A State or StateArray object with the given component added/updated.
        """
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        return self.__class__(x, device=self.device)

    @classmethod
    def from_gym(cls, state, device='cpu', dtype=np.float32):
        """
        Constructs a State object given the return value of an OpenAI gym reset()/step(action) call.

        Args:
            state (tuple): The return value of an OpenAI gym reset()/step(action) call
            device (string): The device on which to store resulting tensors.
            dtype: The type of the observation.

        Returns:
            A State object.
        """
        if not isinstance(state, tuple):
            return State({
                'observation': torch.from_numpy(
                    np.array(
                        state,
                        dtype=dtype
                    ),
                ).to(device)
            }, device=device)

        observation, reward, done, info = state
        observation = torch.from_numpy(
            np.array(
                observation,
                dtype=dtype
            ),
        ).to(device)
        x = {
            'observation': observation,
            'reward': float(reward),
            'done': done,
        }
        info = info if info else {}
        for key in info:
            x[key] = info[key]
        return State(x, device=device)

    def to(self, device):
        if device == self.device:
            return self
        x = {}
        for key, value in self.items():
            if torch.is_tensor(value):
                x[key] = value.to(device)
            else:
                x[key] = value
        return type(self)(x, device=device, shape=self._shape)

    @property
    def observation(self):
        """A tensor containing the current observation."""
        return self['observation']

    @property
    def reward(self):
        """A float representing the reward for the previous state/action pair."""
        return self['reward']

    @property
    def done(self):
        """A boolean that is true if the state is a terminal state, and false otherwise."""
        return self['done']

    @property
    def mask(self):
        """A float that is 1. if the state is non-terminal, or 0. otherwise."""
        return self['mask']

    @property
    def shape(self):
        """The shape of the State or StateArray. A State always has shape ()."""
        return self._shape

    def __len__(self):
        return 1


class StateArray(State):
    """
        An n-dimensional array of environment State objects.

        Internally, all components of the states are represented as n-dimensional tensors.
        This allows for batch-style processing and easy manipulation of states.
        Usually, a StateArray should be constructed using the State.array() function.

        Args:
            x (dict):
                A dictionary containing all state information.
                Each value should be a tensor in which the first n-dimensions
                match the shape of the StateArray.
                The following keys are standard:

                observation (torch.tensor) (required):
                    A tensor representing the observations for each state

                reward (torch.FloatTensor) (optional):
                    A tensor representing rewards for the previous state/action pairs

                done (torch.BoolTensors) (optional):
                    A tensor representing whether each state is terminal

                mask (torch.FloatTensor) (optional):
                    A tensor representing the mask for each state.
            device (string):
                The torch device on which component tensors are stored.
    """

    def __init__(self, x, shape, device='cpu', **kwargs):
        if not isinstance(x, dict):
            x = {'observation': x}
        for k, v in kwargs.items():
            x[k] = v
        if 'observation' not in x:
            raise Exception('StateArray must contain an observation')
        if 'reward' not in x:
            x['reward'] = torch.zeros(shape, device=device)
        if 'done' not in x:
            x['done'] = torch.tensor([False] * np.prod(shape), device=device).view(shape)
        if 'mask' not in x:
            x['mask'] = 1. - x['done'].float()
        super().__init__(x, device=device)
        self._shape = shape

    def update(self, key, value):
        """
        Adds a key/value pair to the StateArray, or updates an existing key/value pair.
        The value should be a tensor whose first n-dimensions match the shape of the StateArray
        Note that this is NOT an in-place operation, but returns a StateArray.

        Args:
            key (string): The name of the state component to update.
            value (any): The value of the new state component.

        Returns:
            A StateArray object with the given component added/updated.
        """
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        return self.__class__(x, self.shape, device=self.device)

    def as_input(self, key):
        value = self[key]
        return value.view((np.prod(self.shape), *value.shape[len(self.shape):])).float()

    def as_output(self, tensor):
        return tensor.view((*self.shape, *tensor.shape[1:]))

    def apply_mask(self, tensor):
        return tensor * self.mask.unsqueeze(-1)

    def flatten(self):
        """
        Converts an n-dimensional StateArray to a 1-dimensional StateArray

        Returns:
            A 1-dimensional StateArray
        """
        n = np.prod(self.shape)
        dims = len(self.shape)
        x = {}
        for k, v in self.items():
            x[k] = v.view((n, *v.shape[dims:]))
        return StateArray(x, (n,), device=self.device)

    def view(self, shape):
        """
        Analogous to torch.tensor.view(), returns a new StateArray object
        containing the same data but with a different shape.

        Returns:
            A StateArray with the given shape
        """
        dims = len(self.shape)
        x = {}
        for k, v in self.items():
            x[k] = v.view((*shape, *v.shape[dims:]))
        return StateArray(x, shape, device=self.device)

    @property
    def observation(self):
        return self['observation']

    @property
    def reward(self):
        return self['reward']

    @property
    def done(self):
        return self['done']

    @property
    def mask(self):
        return self['mask']

    def __getitem__(self, key):
        if isinstance(key, slice):
            shape = self['mask'][key].shape
            return StateArray({k: v[key] for (k, v) in self.items()}, shape, device=self.device)
        if isinstance(key, int):
            return State({k: v[key] for (k, v) in self.items()}, device=self.device)
        if torch.is_tensor(key):
            # some things may get los
            d = {}
            shape = self['mask'][key].shape
            for (k, v) in self.items():
                try:
                    d[k] = v[key]
                except KeyError:
                    pass
            return self.__class__(d, shape, device=self.device)
        try:
            value = super().__getitem__(key)
        except KeyError:
            return None
        return value

    @property
    def shape(self):
        """The shape of the StateArray"""
        return self._shape

    def __len__(self):
        return self.shape[0]


class MultiagentState(State):
    def __init__(self, x, device='cpu', **kwargs):
        if 'agent' not in x:
            raise Exception('MultiagentState must contain an agent ID')
        super().__init__(x, device=device, **kwargs)

    @property
    def agent(self):
        return self['agent']

    @classmethod
    def from_zoo(cls, agent, state, device='cpu', dtype=np.float32):
        """
        Constructs a State object given the return value of an OpenAI gym reset()/step(action) call.

        Args:
            state (tuple): The return value of an OpenAI gym reset()/step(action) call
            device (string): The device on which to store resulting tensors.
            dtype: The type of the observation.

        Returns:
            A State object.
        """
        if not isinstance(state, tuple):
            return MultiagentState({
                'agent': agent,
                'observation': torch.from_numpy(
                    np.array(
                        state,
                        dtype=dtype
                    ),
                ).to(device)
            }, device=device)

        observation, reward, done, info = state
        observation = torch.from_numpy(
            np.array(
                observation,
                dtype=dtype
            ),
        ).to(device)
        x = {
            'agent': agent,
            'observation': observation,
            'reward': float(reward),
            'done': done,
        }
        info = info if info else {}
        for key in info:
            x[key] = info[key]
        return MultiagentState(x, device=device)

    def to(self, device):
        if device == self.device:
            return self
        x = {}
        for key, value in self.items():
            if torch.is_tensor(value):
                x[key] = value.to(device)
            else:
                x[key] = value
        return type(self)(x, device=device, shape=self._shape)
