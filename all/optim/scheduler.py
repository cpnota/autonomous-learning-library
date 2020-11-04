from all.logging import DummyWriter


class Schedulable:
    '''Allow "instance" descriptors to implement parameter scheduling.'''

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if isinstance(value, Scheduler):
            value = value.__get__(self, self.__class__)
        return value


class Scheduler:
    pass


class LinearScheduler(Scheduler):
    def __init__(
            self,
            initial_value,
            final_value,
            decay_start,
            decay_end,
            name='variable',
            writer=DummyWriter(),
    ):
        self._initial_value = initial_value
        self._final_value = final_value
        self._decay_start = decay_start
        self._decay_end = decay_end
        self._i = -1
        self._name = name
        self._writer = writer

    def __get__(self, instance, owner=None):
        result = self._get_value()
        self._writer.add_schedule(self._name, result)
        return result

    def _get_value(self):
        self._i += 1
        if self._i < self._decay_start:
            return self._initial_value
        if self._i >= self._decay_end:
            return self._final_value
        alpha = (self._i - self._decay_start) / (self._decay_end - self._decay_start)
        return alpha * self._final_value + (1 - alpha) * self._initial_value
