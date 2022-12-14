import logging
import random
from matplotlib import pyplot as plt
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner

# VARIABLES FOR EXPERIMENTS:
epochs = 50 # number of epochs to train
penalty_P = 2 # p value for penalization (if relevant)
penalty_function_type = "linear_values" # penalty_function_type = "linear_values"
type_of_mirroring = "weight_value_sizes" # type_of_mirroring = "correlation"
plot_title = "Penalty linear in values, on instances with similar weights and values to the original experiment"

# END VARIABLES FOR EXPERIMENTS

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

data = np.load("./data.npz")
X_1gtrain = data["X_1gtrain"]
X_1gtest = data["X_1gtest"]

# Mirror through weight/value sizes (weights all 3/5/7)
if type_of_mirroring == "weight_value_sizes":
    y_map_function = lambda y: np.round(y / 300) * 2 + 3

    y_train = y_map_function(data["y_train"])
    y_test = y_map_function(data["y_test"])

    X_1gvalidation = X_1gtest[
        0:2880, :
    ]  # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.
    y_validation = y_test[
        0:2880
    ]  # [] floats, length 2880 -> this is the true WEIGHT value of the item. Values around 3,5,7

    y_test = y_test[
        2880:
    ]  # [] floats, length 8496 -> this is the true WEIGHT value of the item. Values around 3,5,7
    X_1gtest = X_1gtest[
        2880:, :
    ]  # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.


    random_values = []
    for i in range(0, 48):
        chance = random.randint(0, 20)
        if chance == 9 or chance == 10:
            random_values.append(np.random.normal(4500, 500))
        elif chance == 8:
            random_values.append(np.random.normal(1000, 200))
        else:
            random_values.append(random.randint(0, 600))
# Mirror through keeping more correlation
else:
    y_map_function = lambda y: np.round(y / 75) + 1

    y_train = y_map_function(data["y_train"])
    y_test = y_map_function(data["y_test"])

    X_1gvalidation = X_1gtest[0:2880, :] # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.
    y_validation = y_test[0:2880] # [] floats, length 2880 -> this is the true WEIGHT value of the item. Values between 1 and 50

    y_test = y_test[2880:] # [] floats, length 8496 -> this is the true WEIGHT value of the item. Values between 1 and 50
    X_1gtest = X_1gtest[2880:, :] # [ 611 0 6 27 7 99 3083.99 53.64 606.13] array of [int int int int int int float float float], 6x int 3x float -> these are the predictive features per item.


    random_values = []
    for i in range(0, 48):
        random_values.append(random.randint(1,70))


clf = MSE_Learner(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title="MSE",
    penalty_P=penalty_P,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title=plot_title,
    plt_show=False,
    penalty_P=2,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title=plot_title,
    plt_show=False,
    penalty_P=10,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title=plot_title,
    plt_show=False,
    penalty_P=100,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title=plot_title,
    plt_show=False,
    penalty_P=1000,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation
)

plt.title(plot_title)
plt.legend(["MSE", "spo_p_2", "spo_p_10", "spo_p_100", "spo_p_1000"])
plt.show()