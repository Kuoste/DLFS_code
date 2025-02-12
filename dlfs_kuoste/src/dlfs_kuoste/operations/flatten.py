import numpy as np

from dlfs_kuoste import operations


class Flatten(operations.Operation):
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_.shape)