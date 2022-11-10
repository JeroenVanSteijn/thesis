from os import listdir
from os.path import isfile, join
from torch import optim
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

# Experiment variables
epochs = 10
nr_training_items = 5520 * 30
nr_validation_items = 2370 * 30
capacity = 300  # 60 is default
n_items = 10  # 48 is default

results_folder = f"./results/energy_{capacity}/"

print(f"running experiments and saving results to folder {results_folder} for {epochs} epochs")

# End experiment variables

# Reading and formatting instance
instance_folder = f"./instances/energy/"
files = [f for f in listdir(instance_folder) if isfile(join(instance_folder, f))]

for index, instance_file in enumerate(files):
    file = open(instance_folder + "/" + instance_file)
    folder = results_folder + str(index) + "/"
    len_items = len(file.readlines()) 
    nr_items = nr_training_items + nr_validation_items # len_items instead if not an instance size experiment.

    x_train, y_train, values_train, x_validation, y_validation, values_validation = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    with open(instance_folder + "/" + instance_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            # Skip header
            if line_count == 0:
                line_count += 1
                continue

            # stop when line_count is the maximum for this experiment.
            if line_count > nr_items:
                break

            line_count += 1

            # Get variables for row
            features = [
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
            ]
            value = float(row[8])
            true_weight = float(row[9])

            if line_count > nr_training_items:
                x_validation.append(features)
                y_validation.append(true_weight)
                values_validation.append(value)

            # training set is all the rest:
            else:
                x_train.append(features)
                y_train.append(true_weight)
                values_train.append(value)

    learner = SGD_SPO_dp_lr(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        n_items=n_items,
        capacity=[capacity],
        penalty_P=1000,
        penalty_function_type="linear_weights",
        file_name=folder + "/spo_learner_p1000_linear_weights.py", # Where to store the results.
    )
    learner.fit(x_train, y_train, x_validation, y_validation)

    learner = MSE_Learner(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        capacity=[capacity],
        n_items=n_items,
        file_name=folder + "/mse_learner.py",  # Where to store the results.
    )
    learner.fit(x_train, y_train, x_validation, y_validation)