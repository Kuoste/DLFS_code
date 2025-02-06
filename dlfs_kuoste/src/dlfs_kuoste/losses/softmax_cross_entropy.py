import numpy as np
from dlfs_kuoste import helpers, losses


class SoftmaxCrossEntropy(losses.Loss):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:

        # if the network is just outputting probabilities
        # of just belonging to one class:
        if self.target.shape[1] == 0:
            self.single_class = True

        # if "single_class", apply the "normalize" operation defined above:
        if self.single_class:
            self.prediction, self.target = helpers.normalize(self.prediction), helpers.normalize(
                self.target
            )

        # applying the softmax function to each row (observation)
        softmax_preds = helpers.softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
            1.0 - self.target
        ) * np.log(1 - self.softmax_preds)

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:

        # if "single_class", "un-normalize" probabilities before returning gradient:
        if self.single_class:
            return helpers.unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]