from dlfs_kuoste import layers, operations
from numpy import ndarray
import numpy as np

class Dense(layers.Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: operations.Operation = operations.Linear(),
                 dropout: float = 1.0,
                 weight_init: str = "standard"):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]
        if self.weight_init == "glorot":
            scale = np.sqrt(2 / (num_in + self.neurons))
        else:
            scale = 1.0

        self.params = []

        # weights
        self.params.append(np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0, scale=scale, size=(1, self.neurons)))


        self.operations = [operations.WeightMultiply(self.params[0]),
                           operations.BiasAdd(self.params[1]),
                           self.activation]
        
        if self.dropout < 1.0: 
            self.operations.append(operations.Dropout(self.dropout))

        return None