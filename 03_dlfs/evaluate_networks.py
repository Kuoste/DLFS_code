from numpy import ndarray
import numpy as np
from dlfs_kuoste import Trainer, NeuralNetwork, losses, operations, optimizers, layers, helpers


def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(helpers.mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(helpers.rmse(preds, y_test)))

lr = NeuralNetwork(
    layers=[layers.Dense(neurons=1,
                   activation=operations.Linear())],
    loss=losses.MeanSquaredError(),
    seed=20190501
)

nn = NeuralNetwork(
    layers=[layers.Dense(neurons=13,
                   activation=operations.Sigmoid()),
            layers.Dense(neurons=1,
                   activation=operations.Linear())],
    loss=losses.MeanSquaredError(),
    seed=20190501
)

dl = NeuralNetwork(
    layers=[layers.Dense(neurons=13,
                   activation=operations.Sigmoid()),
            layers.Dense(neurons=13,
                   activation=operations.Sigmoid()),
            layers.Dense(neurons=1,
                   activation=operations.Linear())],
    loss=losses.MeanSquaredError(),
    seed=20190501
)


USE_BOSTON = False

if USE_BOSTON:
  print("Loading Boston dataset...")
  import pandas as pd
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
  data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  target = raw_df.values[1::2, 2]
  features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
else:
  print("Loading California dataset...")
  from sklearn.datasets import fetch_california_housing
  housing = fetch_california_housing()
  data = housing.data
  target = housing.target
  features = housing.feature_names  # If you need feature names
print("Done, data shape: ", data.shape)

# Scaling the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

def to_2d_np(a: ndarray, 
          type: str="col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

learnrate = 0.001
print("Training linear regression model, learning rate ", learnrate)
trainer = Trainer(lr, optimizers.SGD(lr=learnrate))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(lr, X_test, y_test)
print()

learnrate = 0.01
print("Training neural network model. learning rate ", learnrate)
trainer = Trainer(nn, optimizers.SGD(lr=learnrate))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(nn, X_test, y_test)
print()

learnrate = 0.01
print("Training deep neural network model, learning rate ", learnrate)
trainer = Trainer(dl, optimizers.SGD(lr=learnrate))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(dl, X_test, y_test)
print()