class SchedulerMixin(object):
    '''Change the way attribute getters work to all "instance" descriptors.'''
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if hasattr(value, '__get__'):
            value = value.__get__(self, self.__class__)
        return value


class LinearScheduler(object):
    def __init__(self, initial_value, final_value, decay_start, decay_end):
        self._initial_value = initial_value
        self._final_value = final_value
        self._decay_start = decay_start
        self._decay_end = decay_end
        self._i = -1

    def __get__(self, instance, owner=None):
        self._i += 1
        if self._i < self._decay_start:
            return self._initial_value
        if self._i >= self._decay_end:
            return self._final_value
        alpha = (self._i - self._decay_start) / (self._decay_end - self._decay_start)
        return alpha * self._final_value + (1 - alpha) * self._initial_value
