import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch import nn, optim
from learner import (
    LinearRegression,
    get_kn_indicators,
    get_weights,
    test_fwd,
    train_fwdbwd_mse,
)
import logging
import datetime
from collections import defaultdict


class MSE_Learner:
    def __init__(
        self,
        capacity=None,
        values=None,
        epochs=2,
        plt_show=False,
        doScale=True,
        n_items=48,
        model=None,
        verbose=False,
        plotting=False,
        plot_title="Learning curve",
        optimizer=optim.SGD,
        store_result=False,
        penalty_P=2,
        penalty_function_type="linear_values",
        **hyperparam
    ):
        self.n_items = n_items
        self.capacity = capacity
        self.values = values

        self.hyperparam = hyperparam
        self.epochs = epochs
        self.plot_title = plot_title

        self.plt_show = plt_show

        self.doScale = doScale
        self.verbose = verbose
        self.plotting = plotting
        self.optimizer = optimizer
        self.store_result = store_result

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
        x_test=None,
        y_test=None,
    ):
        x_train = x_train[:, 1:]  # without group ID
        validation = (x_validation is not None) and (y_validation is not None)
        test = (x_test is not None) and (y_test is not None)

        # scale data?
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)

        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        if validation:
            x_validation = x_validation[:, 1:]
            if self.doScale:
                x_validation = self.scaler.transform(x_validation)
            trch_X_validation = torch.from_numpy(x_validation).float()
            trch_y_validation = torch.from_numpy(np.array([y_validation]).T).float()

        if test:
            x_test = x_test[:, 1:]
            if self.doScale:
                x_test = self.scaler.transform(x_test)
            trch_X_test = torch.from_numpy(x_test).float()
            trch_y_test = torch.from_numpy(np.array([y_test]).T).float()

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
        knaps_sol = [
            get_kn_indicators(
                V_true,
                capacity,
                values=self.values,
                true_weights=V_true,
            )
            for V_true in knaps_V_true
        ]
        for k in knaps_sol:
            self.time += k[1]

        if validation:
            n_knapsacks_validation = len(trch_X_validation) // n_items
            knaps_V_true_validation = [
                get_weights(trch_y_validation, kn_nr, n_items)
                for kn_nr in range(n_knapsacks_validation)
            ]
            knaps_sol_validation = [
                get_kn_indicators(
                    V_true,
                    capacity,
                    values=self.values,
                    true_weights=V_true,
                )
                for V_true in knaps_V_true_validation
            ]
            for k in knaps_sol_validation:
                self.time += k[1]

        if test:
            n_knapsacks_test = len(trch_X_test) // n_items
            knaps_V_true_test = [
                get_weights(trch_y_test, kn_nr, n_items)
                for kn_nr in range(n_knapsacks_test)
            ]
            knaps_sol_test = [
                get_kn_indicators(
                    V_true,
                    capacity,
                    values=self.values,
                    true_weights=V_true,
                )
                for V_true in knaps_V_true_test
            ]

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
        logger = []  # (dict_epoch, dict_train, dict_test)
        test_result = []
        knapsack_nrs = [x for x in range(n_knapsacks)]
        for epoch in range(num_epochs):
            logging.info(
                "Training Epoch%d Time:%s\n" % (epoch, str(datetime.datetime.now()))
            )

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
                            values=self.values,
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
                                values=self.values,
                                penalty_P=self.penalty_P,
                                penalty_function_type=self.penalty_function_type
                            )
                        if test:
                            dict_test = test_fwd(
                                self.model,
                                criterion,
                                trch_X_test,
                                trch_y_test,
                                n_items,
                                capacity,
                                knaps_sol_test,
                                values=self.values,
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
                            info["test_loss"] = dict_test["loss"]
                            info["test_regret_full"] = dict_test["regret_full"]
                            info["test_accuracy"] = dict_test["accuracy"]
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
                plt.plot(
                    subepoch_list, regret_list, subepoch_list, regret_list_validation
                )
                plt.title(self.plot_title)
                plt.ylabel("Regret")
                plt.legend(["training", "validation"])
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
