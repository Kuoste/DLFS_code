from torch import Tensor


class Preprocessor():
    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ConvNetPreprocessor(Preprocessor):
    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        # rearrange the dimensions to be (batch, channels, height, width)
        return x.permute(0, 3, 1, 2)
