import logging
from torch import optim
import numpy as np
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="knapsackRunner.log", level=logging.INFO, format=formatter)

# Experiment variables
plot_title = "Penalty function linear in values with P = 10, generated instances"
epochs = 30
penalty_P = 100
penalty_function_type = "linear_weights" # "linear_values"
# End experiment variables

x_train, y_train, values_train, x_validation, y_validation, values_validation = [[],[],[],[],[],[]]

# Reading and formatting instance
instance_file = './instances/1.csv'
file = open(instance_file)
nr_items = len(file.readlines())

with open(instance_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # Skip header
        if line_count == 0:
            line_count += 1
            continue
        line_count += 1
        
        # Get variables for row
        features_ints = [int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])]
        features_floats = [float(row[6]), float(row[7]), float(row[8])]
        features = features_ints + features_floats
        value = int(row[9])
        true_weight = int(row[10])
        # test set is size 75%
        if line_count < nr_items * 0.75:
            x_train.append(features)
            y_train.append(true_weight)
            values_train.append(value)
        # validation set is size 25%
        else:
            x_validation.append(features)
            y_validation.append(true_weight)
            values_validation.append(value)

# Starting learners
learner = MSE_Learner(
    values_train=values_train,
    values_validation=values_validation,
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
run = learner.fit(
    x_train, y_train, x_validation, y_validation
)
print(run.head())

learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
    plot_title=plot_title,
    plt_show=True,
    penalty_P=penalty_P,
    penalty_function_type=penalty_function_type
)
run2 = learner.fit(
    x_train, y_train, x_validation, y_validation
)
print(run2.head())

np.set_printoptions(suppress=True, threshold=1000000)
