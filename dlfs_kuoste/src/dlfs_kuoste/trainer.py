from dlfs_kuoste import NeuralNetwork, optimizers, helpers
import typing
from copy import deepcopy

import numpy as np


class Trainer(object):
    """
    Just a list of layers that runs forwards and backwards
    """

    def __init__(self, net: NeuralNetwork, optim: optimizers.Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, "net", self.net)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
        early_stopping: bool = True,
        conv_testing: bool = False,
    ) -> None:

        setattr(self.optim, "max_epochs", epochs)
        self.optim._setup_decay()

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e + 1) % eval_every == 0:

                last_model = deepcopy(self.net)

            X_train, y_train = helpers.permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

                if conv_testing:
                    if ii % 10 == 0:
                        test_preds = self.net.forward(X_batch, inference = True)
                        batch_loss = self.net.loss.forward(test_preds, y_batch)
                        print("batch", ii, "loss", batch_loss)

                    if ii % 100 == 0 and ii > 0:
                        print(
                            "Validation accuracy after",
                            ii,
                            "batches is",
                            """{0:.2f}%""".format(
                                np.equal(
                                    np.argmax(self.net.forward(X_test), axis=1),
                                    np.argmax(y_test, axis=1),
                                ).sum()
                                * 100.0
                                / X_test.shape[0]
                            ),
                        )

            if (e + 1) % eval_every == 0:

                test_preds = self.net.forward(X_test, inference = True)
                loss = self.net.loss.forward(test_preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print()
                        print(
                            "Loss increased after epoch {0}, final loss was {1:.3f},".format(
                                e + 1, self.best_loss
                            ),
                            "\nusing the model from epoch {0}".format(e + 1 - eval_every),
                        )
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, "net", self.net)
                        break
                else:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()

    def generate_batches(
        self, X: np.ndarray, y: np.ndarray, size: int = 32
    ) -> typing.Generator[typing.Tuple[np.ndarray]]:

        assert (
            X.shape[0] == y.shape[0]
        ), """
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        """.format(
            X.shape[0], y.shape[0]
        )

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch
