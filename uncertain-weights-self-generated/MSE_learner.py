import random
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn, optim
from csv_writer import write_results
from learner import (
    LinearRegression,
    get_kn_indicators,
    get_values,
    get_weights,
    test_fwd,
    train_fwdbwd_mse,
)

class MSE_Learner:
    def __init__(
        self,
        capacity,
        values_train,
        values_validation,
        epochs,
        n_items,
        optimizer,
        penalty_function_type,
        eval_method,
        file_name,
    ):
        self.n_items = n_items
        self.capacity = capacity
        self.values_train = values_train
        self.values_validation = values_validation
        self.epochs = epochs
        self.optimizer = optimizer
        self.best_params_ = {"p": "default"}
        self.penalty_function_type = penalty_function_type
        self.eval_method = eval_method
        self.file_name = file_name

    def fit(
        self,
        x_train,
        y_train,
        x_validation,
        y_validation,
    ):
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        scaler = preprocessing.StandardScaler().fit(x_validation)
        x_validation = scaler.transform(x_validation)
        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        trch_X_validation = torch.from_numpy(x_validation).float()
        trch_y_validation = torch.from_numpy(np.array([y_validation]).T).float()

        # basics
        n_items = self.n_items
        n_knapsacks = len(trch_X_train) // n_items
        capacity = self.capacity

        # prepping
        knaps_V_true = [
            get_weights(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)
        ]
        knaps_values = [
            get_values(self.values_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)
        ]
        
        knaps_sol = [
            get_kn_indicators(
                V_true,
                capacity,
                values=knaps_values[nr]
            )
            for [nr, V_true] in enumerate(knaps_V_true)
        ]

        n_knapsacks_validation = len(trch_X_validation) // n_items
        knaps_values_validation = [
            get_values(self.values_validation, kn_nr, n_items)
            for kn_nr in range(n_knapsacks_validation)
        ]
        knaps_V_true_validation = [
            get_weights(trch_y_validation, kn_nr, n_items)
            for kn_nr in range(n_knapsacks_validation)
        ]
        knaps_sol_validation = [
            get_kn_indicators(
                V_true,
                capacity,
                values=knaps_values_validation[nr]
            )
            for [nr, V_true] in enumerate(knaps_V_true_validation)
        ]

        # network
        self.model = LinearRegression(
            trch_X_train.shape[1], 1
        )  # input dim, output dim

        # loss
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters())
        num_epochs = self.epochs

        # training
        subepoch = 0  # for logging and nice curves
        test_result = []
        knapsack_nrs = [x for x in range(n_knapsacks)]
        for epoch_nr in range(num_epochs):
            print(epoch_nr)
            random.shuffle(knapsack_nrs)  # randomly shuffle order of training
            cnt = 0
            for kn_nr in knapsack_nrs:
                kn_start = kn_nr * n_items
                kn_stop = kn_start + n_items

                train_fwdbwd_mse(
                    self.model,
                    optimizer,
                    trch_X_train[kn_start:kn_stop],
                    trch_y_train[kn_start:kn_stop],
                )

                cnt += 1
                subepoch += 1
                if cnt % 20 == 0:
                    dict_validation = test_fwd(
                        self.model,
                        criterion,
                        trch_X_validation,
                        trch_y_validation,
                        n_items,
                        capacity,
                        knaps_sol_validation,
                        values=self.values_validation,
                        eval_method=self.eval_method
                    )

                    info = {}
                    info["validation_loss"] = dict_validation["loss"]
                    info["validation_regret_full"] = dict_validation["regret_full"]
                    info["validation_accuracy"] = dict_validation["accuracy"]
                    info["subepoch"] = subepoch
                    info["epoch_nr"] = epoch_nr
                    test_result.append(info)
        write_results(self.file_name, test_result)