from typing import List
from numpy import ndarray
import numpy as np
from dlfs_kuoste import layers, losses

class NeuralNetwork(object):
    '''
    The class for a neural network.
    '''
    def __init__(self, 
                 layers: List[layers.Layer],
                 loss: losses.Loss,
                 seed: int = 1) -> None:
        '''
        Neural networks need layers, and a loss.
        '''
        self.layers = layers
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

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
        Passes data forward through a series of layers.
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
        Passes data backward through a series of layers.
        '''

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:
        '''
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        '''
        
        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss
    
    def params(self):
        '''
        Gets the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Gets the gradient of the loss with respect to the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.param_grads

            