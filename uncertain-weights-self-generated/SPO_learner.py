import random
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
from csv_writer import write_results
from learner import LinearRegression, get_kn_indicators, get_objective_value_penalized_infeasibility, get_values, get_weights, get_weights_pred, train_fwdbwd_grad, test_fwd

class SGD_SPO_dp_lr:
    def __init__(
        self,
        capacity,
        values_train,
        values_validation,
        epochs,
        n_items,
        optimizer,
        penalty_P,
        penalty_function_type,
        file_name,
    ):
        self.n_items = n_items
        self.capacity = capacity
        self.values_train = values_train
        self.values_validation = values_validation
        self.epochs = epochs
        self.optimizer = optimizer
        self.best_params_ = {"p": "default"}
        self.penalty_P = penalty_P
        self.penalty_function_type = penalty_function_type
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
                V_true = knaps_V_true[kn_nr]
                sol_true = knaps_sol[kn_nr][0]
                values_instance = knaps_values[kn_nr]

                # SPO+ gradient
                V_pred = get_weights_pred(self.model, trch_X_train, kn_nr, n_items)
                V_spo = 2 * V_pred - V_true
                
                assignments_spo, t = get_kn_indicators(
                    V_spo,
                    capacity,
                    values=values_instance
                )
                # Objective value for 2 * theta hat - theta
                sol_spo, _was_penalized = get_objective_value_penalized_infeasibility(assignments_spo, V_true, values_instance, capacity, self.penalty_P, self.penalty_function_type)
                grad = (sol_spo - sol_true)
                
                ### what if for the whole 48 items at a time
                kn_start = kn_nr * n_items
                kn_stop = kn_start + n_items
                train_fwdbwd_grad(
                    self.model,
                    optimizer,
                    trch_X_train[kn_start:kn_stop],
                    trch_y_train[kn_start:kn_stop],
                    torch.from_numpy(np.array([grad]).T).float(),
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
                        values=self.values_validation
                    )

                    info = {}
                    info["validation_regret_full_linear_values"] = dict_validation[
                        "regret_full_linear_values"
                    ]
                    info["validation_regret_full_rejection"] = dict_validation[
                        "regret_full_rejection"
                    ]
                    info["epoch_nr"] = epoch_nr
                    test_result.append(info)
        write_results(self.file_name, test_result)