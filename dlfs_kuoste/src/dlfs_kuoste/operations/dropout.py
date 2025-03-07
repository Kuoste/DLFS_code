from numpy import ndarray
from dlfs_kuoste import operations
import numpy as np

class Dropout(operations.Operation):
    def __init__(self, keep_prob: float = 0.8): 
        super().__init__() 
        self.keep_prob = keep_prob

    def _output(self, inference: bool = False) -> ndarray: 
        if inference: 
              return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray: 
        return output_grad * self.mask