from dlfs_kuoste import layers
from dlfs_kuoste import operations
import numpy as np

class Conv2D(layers.Layer):
    """
    Once we define all the Operations and the outline of a layer,
    all that remains to implement here is the _setup_layer function!
    """

    def __init__(
        self,
        out_channels: int,
        param_size: int,
        dropout: int = 1.0,
        weight_init: str = "normal",
        activation: operations.Operation = operations.Linear(),
        flatten: bool = False,
    ) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels

    def _setup_layer(self, input_: np.ndarray) -> np.ndarray:

        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2 / (in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(
            loc=0,
            scale=scale,
            size=(
                input_.shape[1],  # input channels
                self.out_channels,
                self.param_size,
                self.param_size,
            ),
        )

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(operations.Conv2D(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(operations.Flatten())

        if self.dropout < 1.0:
            self.operations.append(operations.Dropout(self.dropout))

        return None