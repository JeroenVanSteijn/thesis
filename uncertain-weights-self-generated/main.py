from torch import optim
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

# Experiment variables
epochs = 100
penalty_function_type = "linear_weights" # "linear_values"
folder = "./results/01-07_no_noise_multi_p_linear_weights"

# End experiment variables

x_train, y_train, values_train, x_validation, y_validation, values_validation = [[],[],[],[],[],[]]

# Reading and formatting instance
instance_file = './instances/0.1_noise.csv'
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

        # training set is size 75%
        if line_count < nr_items * 0.75:
            x_train.append(features)
            y_train.append(true_weight)
            values_train.append(value)

        # validation set is size 25%
        else:
            x_validation.append(features)
            y_validation.append(true_weight)
            values_validation.append(value)

# Running learners
learner = MSE_Learner(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    capacity=[60],
    n_items=48,
    penalty_P=1,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder + "/mse_learner.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)

learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    n_items=48,
    capacity=[60],
    penalty_P=1,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder+"/spo_learner_p1.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)

learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    n_items=48,
    capacity=[60],
    penalty_P=2,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder+"/spo_learner_p2.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)

learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    n_items=48,
    capacity=[60],
    penalty_P=10,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder+"/spo_learner_p10.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)


learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    n_items=48,
    capacity=[60],
    penalty_P=100,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder+"/spo_learner_p100.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)


learner = SGD_SPO_dp_lr(
    values_train=values_train,
    values_validation=values_validation,
    epochs=epochs,
    optimizer=optim.Adam,
    n_items=48,
    capacity=[60],
    penalty_P=100,
    penalty_function_type=penalty_function_type,
    repair_in_validation=True,
    file_name=folder+"/spo_learner_p1000.py",
)
learner.fit(
    x_train, y_train, x_validation, y_validation
)
