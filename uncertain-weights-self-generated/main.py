from os import listdir
from os.path import isfile, join
from torch import optim
from SPO_learner import SGD_SPO_dp_lr
from MSE_learner import MSE_Learner
import csv

# Experiment variables
epochs = 100
penalty_function_type = "linear_values" # "linear_weights"
results_folder = "./results/linear_combination_no_noise/"

# End experiment variables
capacity = 60 # 60 is default
n_items = 48 # 48 is default

x_train, y_train, values_train, x_validation, y_validation, values_validation = [[],[],[],[],[],[]]

# Reading and formatting instance
instance_folder = './instances/linear_combination_no_noise'
files = [f for f in listdir(instance_folder) if isfile(join(instance_folder, f))]

for index, instance_file in enumerate(files):
    file = open(instance_folder + '/' + instance_file)
    folder = results_folder + str(index) + '/'
    nr_items = len(file.readlines())

    with open(instance_folder + '/' + instance_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # Skip header
            if line_count == 0:
                line_count += 1
                continue
            line_count += 1
            
            # Get variables for row
            features_ints = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
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
    learner = SGD_SPO_dp_lr(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        n_items=n_items,
        capacity=[capacity],
        penalty_P=10,
        penalty_function_type=penalty_function_type,
        reject_in_validation=True,
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
        n_items=n_items,
        capacity=[capacity],
        penalty_P=1000,
        penalty_function_type=penalty_function_type,
        reject_in_validation=True,
        file_name=folder+"/spo_learner_p1000.py",
    )
    learner.fit(
        x_train, y_train, x_validation, y_validation
    )

    learner = MSE_Learner(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        capacity=[capacity],
        n_items=n_items,
        penalty_P=1,
        penalty_function_type=penalty_function_type,
        reject_in_validation=True,
        file_name=folder + "/mse_learner.py",
    )
    learner.fit(
        x_train, y_train, x_validation, y_validation
    )