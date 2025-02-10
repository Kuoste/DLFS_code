from dlfs_kuoste import optimizers
import numpy as np

class SgdMomentum(optimizers.Optimizer):
    def __init__(
        self, lr: float = 0.01, final_lr: float = 0, decay_type: optimizers.DecayType = None, momentum: float = 0.9
    ) -> None:
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum
        self.first = True

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for param, param_grad, velocity in zip(
            self.net.params(), self.net.param_grads(), self.velocities
        ):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

        # Update velocity
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]

        # Use this to update parameters
        kwargs["param"] -= kwargs["velocity"]