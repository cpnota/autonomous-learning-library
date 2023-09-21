import gym
import torch
import numpy as np
from . import training_utils
from all.core import State
from .duplicate_env import DuplicateEnvironment
from ._environment import Environment


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)


def load_data(name):
    training_data = np.loadtxt(f'{name}-training_data.txt')
    training_classes = np.loadtxt(f'{name}-training_classes.txt')
    test_data = np.loadtxt(f'{name}-test_data.txt')
    test_classes = np.loadtxt(f'{name}-test_classes.txt')
    return training_data, training_classes, test_data, test_classes


def apply_gates_classical(angles):
    qubit = np.array([np.sin(angles[0]),0,np.cos(angles[0])])
    for i in range(1,len(angles)-1):
        angle = angles[i]
        cos = np.cos(angle)
        sin = np.sin(angle)
        if i % 2 == 0:
            rotation = np.array([[cos, 0, -sin],
                                 [0, 1, 0],
                                 [sin, 0, cos]])
        else:
            rotation = np.array([[cos, sin, 0],
                                 [-sin, cos, 0],
                                 [0, 0, 1]])
        qubit = qubit @ rotation
    z = qubit[-1] # z in [-1, 1]. 1 = |0> and -1 = |1>
    value = 1 - (z+1)/2
    return value


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration >= total:
        print()


class QlassifierEnvironment(Environment):
    def __init__(self, name, device='cpu', param_range=(-2,5), threshold=0.99, max_iterations=250):

        # load data
        self.n_layers = int(name[-8])
        self.ansatz = name[-11:-9]
        self.training_data, self.training_classes, self.test_data, self.test_classes = load_data(name)
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.n_states = self.n_layers * 4
        self.step_size = [0.5, 0.05]
        self.n_actions = self.n_layers * 4 * len(self.step_size) * 2 + 1
        self.accuracies_and_losses = {}
        self.param_lo, self.param_hi = param_range

        # initialize member variables
        self._name = name
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device


    def reset(self):
        state = torch.FloatTensor(self.n_layers*4,).uniform_(self.param_lo, self.param_hi)
        self._state = State(state, self._device)
        self.best_accuracy = 0
        self.iteration = 0
        self.best_loss = float("inf")
        print()
        accuracy, loss = self.measure(state)
        self.initial_accuracy = accuracy
        self.initial_loss = loss
        return self._state


    def measure(self, param):
        if str(param) in self.accuracies_and_losses:
            accuracy, loss = self.accuracies_and_losses[str(param)]
        else:
            angles = training_utils.generate_angles(self.training_data, param, self.ansatz)
            measurements = np.array([apply_gates_classical(angle) for angle in angles])
            accuracy = ((measurements >= 0.5) == self.training_classes).sum() / len(measurements)
            loss = binary_cross_entropy(self.training_classes, measurements)
            self.accuracies_and_losses[str(param)] = (accuracy, loss)
            self.iteration += 1
            if accuracy > self.best_accuracy or loss < self.best_loss:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                if loss < self.best_loss:
                    self.best_loss = loss
            suffix = f'accuracy (max): {round(accuracy*1000)/10}% ({round(self.best_accuracy*1000)/10}%)'
            suffix += f' loss (min): {round(loss*1000)/1000} ({round(self.best_loss*1000)/1000})'
            print_progress_bar(self.iteration, self.max_iterations, suffix=suffix, length=20, fill = '*')
        return accuracy, loss


    def step(self, action):
        if action == self.n_actions - 1:
            state = torch.FloatTensor(self.n_layers*4,).uniform_(self.param_lo, self.param_hi)
        else:
            '''
                Action: 0 1 2 3 4 5 | 6 7 8 9 10 11 | ...
                Param:     param1   |    param2     | ...
                Step_s:  0 . 1 . 2  |  0 . 1 .  2   | ...
                Sign:   + - + - + - | + - + -  +  - | ... 
            '''
            param_to_act = action // (len(self.step_size) * 2)
            action_index = action % (len(self.step_size) * 2)
            action_step_size = self.step_size[action_index // len(self.step_size)]
            action_sign = 1 if action_index % 2 == 0 else -1
            delta = np.random.random() * action_step_size * action_sign

            old_state = self._state.observation
            state = old_state.detach().clone()
            state[param_to_act] += delta

        old_best_accuracy = self.best_accuracy
        new_accuracy, new_loss = self.measure(state)
        accuracy_gain = new_accuracy - old_best_accuracy
        max_accuracy_gain = 1 - self.initial_accuracy
        reward = max(0, accuracy_gain / max_accuracy_gain) * 100
        done = new_accuracy > self.threshold or self.iteration >= self.max_iterations

        self._state = State({'observation': state,
                             'reward': reward,
                             'done': done},
                            self._device)
        return self._state

    def render(self, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed):
        np.seed(seed)

    def duplicate(self, n):
        return DuplicateEnvironment([QlassifierEnvironment(self._name, device=self._device) for _ in range(n)])

    @property
    def name(self):
        return self._name

    @property
    def state_space(self):
        breakpoint()
        return self._env.observation_space

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(self.n_actions)

    @property
    def state(self):
        return self._state

    @property
    def env(self):
        breakpoint()
        return self._env

    @property
    def device(self):
        return self._device
