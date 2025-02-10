from dlfs_kuoste import optimizers

class Sgd(optimizers.Optimizer):
    def __init__(self, lr: float = 0.01, final_lr: float = 0, decay_type: optimizers.DecayType = None) -> None:
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:

        update = self.lr * kwargs["grad"]
        kwargs["param"] -= update