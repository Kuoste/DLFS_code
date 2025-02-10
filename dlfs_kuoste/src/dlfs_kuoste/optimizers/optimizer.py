from enum import Enum
import numpy as np

# Note to self, in python enums seem to be a world of their own:
# https://docs.python.org/3/howto/enum.html#enum-class-differences
class DecayType(Enum):
    LINEAR = 1
    EXPONENTIAL = 2

# DecayType = Enum('DecayType', [('LINEAR', 1), ('EXPONENTIAL', 2)])

class Optimizer(object):
    def __init__(self, lr: float = 0.01, final_lr: float = 0, dt: DecayType = None) -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.dt = dt
        self.first = True

    def _setup_decay(self) -> None:

        if not self.dt:
            return
        
        if self.dt == DecayType.EXPONENTIAL:
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (self.max_epochs - 1))
        elif self.dt == DecayType.LINEAR:
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:

        if not self.dt:
            return
        
        if self.dt == DecayType.EXPONENTIAL:
            self.lr *= self.decay_per_epoch
        elif self.dt == DecayType.LINEAR:
            self.lr -= self.decay_per_epoch

    def step(self) -> None:

        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()