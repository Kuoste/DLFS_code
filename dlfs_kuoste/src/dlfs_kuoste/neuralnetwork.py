import typing
import numpy as np

from dlfs_kuoste import layers, losses


class LayerBlock(object):

    def __init__(self, layers: typing.List[layers.Layer]):
        super().__init__()
        self.layers = layers

    def forward(self, X_batch: np.ndarray, inference: bool = False) -> np.ndarray:

        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    """
    Just a list of layers that runs forwards and backwards
    """

    def __init__(
        self,
        layers: typing.List[layers.Layer],
        loss: losses.Loss = losses.MeanSquaredError,
        seed: int = 1,
    ):
        super().__init__(layers)
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def __str__(self) -> str:
        '''
        Returns the network specifications as a string.
        '''
        t = "    " # indent
        layers_str = f"{t}layers=[\n"
        for layer in self.layers:
            layers_str += f"{t}{t}{layer.__class__.__name__}(neurons={layer.neurons}, activation={layer.activation.__class__.__name__}()),\n"
        layers_str += f"{t}],\n"

        loss_params = ""
        if hasattr(self.loss, 'normalize'):
            loss_params += f"normalize={getattr(self.loss, 'normalize')}"

        loss_str = f"{t}loss={self.loss.__class__.__name__}({loss_params}),\n"
        seed_str = f"{t}seed={self.seed},"
        return layers_str + loss_str + seed_str

    def forward_loss(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:

        prediction = self.forward(X_batch)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:

        prediction = self.forward(X_batch)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss
