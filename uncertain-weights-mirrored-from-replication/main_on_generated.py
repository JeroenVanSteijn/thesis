import logging
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

data = []
with open('./iunstances/1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        if line_count > 0:
            features = row[:9]
            


epochs = 30
penalty_P = 100
# penalty_function_type = "linear_values"
penalty_function_type = "linear_weights"

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
    X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test
)

clf = SGD_SPO_dp_lr(
    values=random_values,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title="Penalty function linear in weights with P = 10",
    plt_show=True,
    penalty_P=penalty_P,
    penalty_function_type=penalty_function_type
)
pdf = clf.fit(
    X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test
)
print(pdf.head())

np.set_printoptions(suppress=True, threshold=1000000)
print(max(y_validation))
