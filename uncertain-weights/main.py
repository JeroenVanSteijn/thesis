import logging
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

x_train, y_train, x_validation, y_validation = []

# Read instance
with open('./instances/1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    nr_items = len(csv_reader)
    for row in csv_reader:
        line_count += 1
        
        if line_count == 0:
            # Skip header
            break

        features = row[:9]
        value = row[10]
        true_weight = row[11]
        if line_count < nr_items * 0.75: # test set is size 75%
            x_train.append(features)
            y_train.append(true_weight)
        else: # validation set is size 25%
            x_validation.append(features)
            y_train.append(true_weight)


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
    x_train, y_train, x_validation, y_validation
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
    x_train, y_train, X_1gvalidation, y_validation
)
print(pdf.head())

np.set_printoptions(suppress=True, threshold=1000000)
print(max(y_validation))
