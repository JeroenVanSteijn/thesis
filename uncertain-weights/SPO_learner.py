import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch import nn, optim
from learner import LinearRegression, get_kn_indicators, get_objective_value_penalized_infeasibility, get_values, get_weights, get_weights_pred, train_fwdbwd_grad, test_fwd
import logging
import datetime
from collections import defaultdict

class SGD_SPO_dp_lr:
    def __init__(
        self,
        capacity=None,
        values_train=None,
        values_validation=None,
        epochs=2,
        doScale=True,
        n_items=48,
        model=None,
        verbose=False,
        plotting=False,
        plt_show=False,
        plot_title="Learning curve",
        optimizer=optim.SGD,
        store_result=False,
        penalty_P=2,
        penalty_function_type="linear_values",
        **hyperparam
    ):
        self.n_items = n_items
        self.capacity = capacity
        self.values_train = values_train
        self.values_validation = values_validation

        self.hyperparam = hyperparam
        self.epochs = epochs
        self.plot_title = plot_title

        self.doScale = doScale
        self.verbose = verbose
        self.plotting = plotting
        self.optimizer = optimizer
        self.store_result = store_result
        self.plt_show = plt_show

        self.scaler = None
        self.model = model
        self.best_params_ = {"p": "default"}
        self.time = 0

        self.penalty_P = penalty_P
        self.penalty_function_type = penalty_function_type

    def fit(
        self,
        x_train,
        y_train,
        x_validation=None,
        y_validation=None,
    ):
        validation = (x_validation is not None) and (y_validation is not None)

        # scale data?
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)

        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        if validation:
            if self.doScale:
                x_validation = self.scaler.transform(x_validation)
            trch_X_validation = torch.from_numpy(x_validation).float()
            trch_y_validation = torch.from_numpy(np.array([y_validation]).T).float()

        if self.plotting:
            subepoch_list = []
            loss_list = []
            regret_list = []
            accuracy_list = []
            if validation:
                loss_list_validation = []
                regret_list_validation = []
                accuracy_list_validation = []

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
                values=knaps_values[nr],
                true_weights=V_true,
            )
            for [nr, V_true] in enumerate(knaps_V_true)
        ]
        for k in knaps_sol:
            self.time += k[1]

        if validation:
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
                    values=knaps_values_validation[nr],
                    true_weights=V_true,
                )
                for [nr, V_true] in enumerate(knaps_V_true_validation)
            ]
            for k in knaps_sol_validation:
                self.time += k[1]

        # network
        if not self.model:
            self.model = LinearRegression(
                trch_X_train.shape[1], 1
            )  # input dim, output dim

        # loss
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), **self.hyperparam)
        num_epochs = self.epochs

        # training
        subepoch = 0  # for logging and nice curves
        logger = []  # (dict_epoch, dict_train)
        test_result = []
        knapsack_nrs = [x for x in range(n_knapsacks)]
        for epoch in range(num_epochs):
            logging.info(
                "Training Epoch%d Time:%s\n" % (epoch, str(datetime.datetime.now()))
            )

            random.shuffle(knapsack_nrs)  # randomly shuffle order of training
            cnt = 0
            enable_logging = cnt % 20 == 0 and False
            for kn_nr in knapsack_nrs:
                V_true = knaps_V_true[kn_nr]
                sol_true = knaps_sol[kn_nr][0]
                values_instance = knaps_values[kn_nr]
                optimal_objective_value = np.sum(sol_true * values_instance)

                # the true-shifted predictions
                V_pred = get_weights_pred(self.model, trch_X_train, kn_nr, n_items)
                V_spo = 2 * V_pred - V_true

                assignments_pred, t = get_kn_indicators(
                    V_pred,
                    capacity,
                    values=values_instance,
                    true_weights=V_true,
                    warmstart=sol_true,
                    logging=enable_logging
                )
                # Objective value for theta hat
                sol_pred, _was_penalized = get_objective_value_penalized_infeasibility(assignments_pred, V_true, values_instance, capacity, self.penalty_P, self.penalty_function_type)

                assignments_spo, t = get_kn_indicators(
                    V_spo,
                    capacity,
                    values=values_instance,
                    true_weights=V_true,
                    warmstart=sol_true,
                    logging=enable_logging
                )
                # Objective value for 2 * theta hat - theta
                sol_spo, _was_penalized = get_objective_value_penalized_infeasibility(assignments_spo, V_true, values_instance, capacity, self.penalty_P, self.penalty_function_type)
                
                regret = optimal_objective_value - sol_pred

                grad = regret * (sol_spo - sol_true)
                
                self.time += t

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

                if self.verbose or self.plotting or self.store_result:
                    cnt += 1
                    subepoch += 1
                    if cnt % 20 == 0:
                        dict_epoch = {
                            "epoch": epoch + 1,
                            "subepoch": subepoch,
                            "cnt": cnt,
                        }
                        dict_train = test_fwd(
                            self.model,
                            criterion,
                            trch_X_train,
                            trch_y_train,
                            n_items,
                            capacity,
                            knaps_sol,
                            values=self.values_train,
                            penalty_P=self.penalty_P,
                            penalty_function_type=self.penalty_function_type
                        )
                        if validation:
                            dict_validation = test_fwd(
                                self.model,
                                criterion,
                                trch_X_validation,
                                trch_y_validation,
                                n_items,
                                capacity,
                                knaps_sol_validation,
                                values=self.values_validation,
                                penalty_P=self.penalty_P,
                                penalty_function_type=self.penalty_function_type
                            )
                        self.time += dict_validation["runtime"]
                        if self.store_result:
                            info = {}
                            info["train_loss"] = dict_train["loss"]
                            info["train_regret_full"] = dict_train["regret_full"]
                            info["train_accuracy"] = dict_train["accuracy"]
                            info["validation_loss"] = dict_validation["loss"]
                            info["validation_regret_full"] = dict_validation[
                                "regret_full"
                            ]
                            info["validation_accuracy"] = dict_validation["accuracy"]
                            info["subepoch"] = subepoch
                            info["time"] = self.time
                            test_result.append(info)

                        if self.plotting:
                            loss_list.append(dict_train["loss"])
                            regret_list.append(dict_train["regret_full"])
                            accuracy_list.append(
                                (dict_train["tn"] + dict_train["tp"])
                                / (
                                    dict_train["tn"]
                                    + dict_train["tp"]
                                    + dict_train["fp"]
                                    + dict_train["fn"]
                                )
                            )
                            subepoch_list.append(subepoch)
                            if validation:
                                loss_list_validation.append(dict_validation["loss"])
                                regret_list_validation.append(
                                    dict_validation["regret_full"]
                                )
                                accuracy_list_validation.append(
                                    (dict_validation["tn"] + dict_validation["tp"])
                                    / (
                                        dict_validation["tn"]
                                        + dict_validation["tp"]
                                        + dict_validation["fp"]
                                        + dict_validation["fp"]
                                    )
                                )
                        if self.verbose:
                            if validation:
                                logger.append((dict_epoch, dict_train, dict_validation))
                                print(
                                    "Epoch[{}/{}]::{}, loss(train): {:.6f}, regret(train): {:.2f},  penalized(train): {:.2f}, loss(validation): {:.6f}, regret(validation): {:.2f}, penalized(train): {:.2f}".format(
                                        epoch + 1,
                                        num_epochs,
                                        cnt,
                                        dict_train["loss"],
                                        dict_train["regret_full"],
                                        dict_train["penalized_count"],
                                        dict_validation["loss"],
                                        dict_validation["regret_full"],
                                        dict_validation["penalized_count"],
                                    )
                                )
                            else:
                                logger.append((dict_epoch, dict_train))
                                print(
                                    "Epoch[{}/{}]::{}, loss: {:.6f}, regret(train): {:.2f}".format(
                                        epoch + 1,
                                        num_epochs,
                                        cnt,
                                        dict_train["loss"],
                                        dict_train["regret_full"],
                                    )
                                )

        if self.plotting:

            if validation:
                legend = ["training MSE", "validation MSE", "training SPO", "validation SPO"] if self.plt_show else ["training", "validation"] 
                plt.plot(
                    subepoch_list, regret_list, subepoch_list, regret_list_validation
                )
                plt.title(self.plot_title)
                plt.ylabel("Regret")
                plt.legend(legend)
                plt.xlabel("Sub Epochs")
                if self.plt_show:
                    plt.show()
                else:
                    plt.savefig("mse_plot.png")
            else:
                plt.plot(
                    subepoch_list, regret_list, subepoch_list, regret_list
                )
                plt.title(self.plot_title)
                plt.ylabel("Regret")
                plt.legend(["training", "validation"])
                plt.xlabel("Sub Epochs")
                if self.plt_show:
                    plt.show()
                else:
                    plt.savefig("mse_plot.png")

        if self.store_result:
            dd = defaultdict(list)
            for d in test_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            logging.info("Completion Time %s \n" % str(datetime.datetime.now()))
            return df
