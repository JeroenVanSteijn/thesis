import logging
import random
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

data = np.load("./data.npz")
X_1gtrain = data["X_1gtrain"]
X_1gtest = data["X_1gtest"]

y_map_function = lambda y: np.round(y / 75) + 1

y_train = y_map_function(data["y_train"])
y_test = y_map_function(data["y_test"])

X_1gvalidation = X_1gtest[
    0:2880, :
]  # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.
y_validation = y_test[
    0:2880
]  # [] floats, length 2880 -> this is the true WEIGHT value of the item. Values between 1 and 50

y_test = y_test[
    2880:
]  # [] floats, length 8496 -> this is the true WEIGHT value of the item. Values between 1 and 50
X_1gtest = X_1gtest[
    2880:, :
]  # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.


random_values = []
for i in range(0, 48):
    random_values.append(random.randint(1, 70))

clf = MSE_Learner(
    values=random_values,
    epochs=30,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title="MSE"
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=30,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title="SPO vs MSE",
    plt_show=True
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test
)
print(pdf.head())

# np.set_printoptions(suppress=True, threshold=1000000)
# print(max(y_validation))
