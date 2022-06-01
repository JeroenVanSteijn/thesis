import logging
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

data = np.load("./data.npz")
X_1gtrain = data["X_1gtrain"]
X_1gtest = data["X_1gtest"]
y_train = data["y_train"]
y_test = data["y_test"]
X_1gvalidation = X_1gtest[0:2880, :] # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.
y_validation = y_test[0:2880] # [] floats, length 2880 -> this is the true value of the item. Values between 0.1 and 3642.74
y_test = y_test[2880:] # [] floats, length 8496 -> this is the true value of the item. Values between 0.1 and 3642.74
X_1gtest = X_1gtest[2880:, :] # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.
weights = [data["weights"].tolist()] # [[5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3, 5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3, 7, 7, 3, 7, 5, 5, 7, 3, 7, 3, 3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7]]
weights = np.array(weights)

clf = SGD_SPO_dp_lr(
    weights=weights,
    epochs=10,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
)
pdf = clf.fit(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test)
print(pdf.head())
