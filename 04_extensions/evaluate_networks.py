import numpy as np

from dlfs_kuoste import operations
from dlfs_kuoste import layers
from dlfs_kuoste import losses
from dlfs_kuoste import optimizers
from dlfs_kuoste import helpers
from dlfs_kuoste import NeuralNetwork
from dlfs_kuoste import Trainer

RANDOM_SEED = 190119

# _train arrays contain 28x28 pixel images, one image per row
# _test arrays contain labels for the images
X_train, y_train, X_test, y_test = helpers.mnist_load()

num_labels = len(y_train)
print("Training data size: ", num_labels)

# Labels get ones in the corresponding position of the number (one-hot encode)
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1
num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

# Scale data to mean 0 and variance 1
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
# np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)
# np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)

def calc_accuracy_model(model, test_set):
    return print(
        '''The model validation accuracy is: {0:.2f}%'''.format(
            np.equal(np.argmax(model.forward(test_set), axis=1), y_test).sum()
            * 100.0
            / test_set.shape[0]
        )
    )


model = NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=operations.Sigmoid()),
        layers.Dense(neurons=10, activation=operations.Sigmoid()),
    ],
    loss=losses.MeanSquaredError(normalize=False),
    seed=RANDOM_SEED,
)
print("\nTrain a model with sigmoid activations:")
print(model)

trainer = Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train, train_labels,
    X_test, test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)
calc_accuracy_model(model, X_test)


model = NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=operations.Sigmoid()),
        layers.Dense(neurons=10, activation=operations.Linear()),
    ],
    loss=losses.MeanSquaredError(normalize=True),
    seed=RANDOM_SEED,
)
print("\nTry turning on normalization on mse ",
      "and change the activation on the last layer to Linear:")
print(model)

trainer = Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train, train_labels,
    X_test, test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)
calc_accuracy_model(model, X_test)

model = NeuralNetwork(
    layers=[
        layers.Dense(neurons=89, activation=operations.Sigmoid()),
        layers.Dense(neurons=10, activation=operations.Linear()),
    ],
    loss=losses.SoftmaxCrossEntropy(),
    seed=RANDOM_SEED,
)
print("\nBetter but still no good. Try using softmax:")
print(model)

trainer = Trainer(model, optimizers.SGD(0.1))
trainer.fit(
    X_train, train_labels,
    X_test, test_labels,
    epochs=50,
    eval_every=5,
    seed=RANDOM_SEED,
    batch_size=60,
)
calc_accuracy_model(model, X_test)
