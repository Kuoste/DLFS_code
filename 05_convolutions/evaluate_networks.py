from dlfs_kuoste import helpers
from dlfs_kuoste import layers
from dlfs_kuoste import losses
from dlfs_kuoste import optimizers
from dlfs_kuoste import operations
from dlfs_kuoste import NeuralNetwork
from dlfs_kuoste import Trainer

import numpy as np


X_train, y_train, X_test, y_test = helpers.mnist_load()

X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

X_train_conv, X_test_conv = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)

num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: 
    {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

model = NeuralNetwork(
    layers=[layers.Conv2D(out_channels=16,
                   param_size=5,
                   dropout=0.8,
                   weight_init="glorot",
                   flatten=True,
                  activation=operations.Tanh()),
            layers.Dense(neurons=10, 
                  activation=operations.Linear())],
            loss = losses.SoftmaxCrossEntropy(), 
seed=20190402)

print("Model is:")
print(model)
print("\nTraining the model for 1 epoch with batch size 60...\n")

trainer = Trainer(model, optimizers.SgdMomentum(lr = 0.1, momentum=0.9))
trainer.fit(X_train_conv, train_labels, X_test_conv, test_labels,
            epochs = 1,
            eval_every = 1,
            seed=20190402,
            batch_size=60,
            conv_testing=True);

calc_accuracy_model(model, X_test_conv)