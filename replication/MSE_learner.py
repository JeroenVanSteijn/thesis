import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch import nn, optim
from torch.autograd import Variable
from learner import (
    LinearRegression,
    get_kn_indicators,
    get_profits,
    get_profits_pred,
    train_prediction_error,
    test_fwd,
)
import logging
import datetime
from collections import defaultdict


class MSE_learner:
    def __init__(
        self,
        capacity=None,
        weights=None,
        epochs=2,
        doScale=True,
        n_items=48,
        model=None,
        verbose=False,
        plotting=False,
        return_regret=False,
        validation_relax=False,
        degree=1,
        optimizer=optim.SGD,
        store_result=False,
        **hyperparam
    ):
        self.n_items = n_items
        self.capacity = capacity
        self.weights = weights

        self.hyperparam = hyperparam
        self.epochs = epochs

        self.doScale = doScale
        self.verbose = verbose
        self.plotting = plotting
        self.return_regret = return_regret
        self.optimizer = optimizer
        self.degree = degree
        self.validation_relax = validation_relax
        self.store_result = store_result

        self.scaler = None
        self.model = model
        self.best_params_ = {"p": "default"}
        self.time = 0

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
            if test:
                regret_list_test = []

        # basics
        n_items = self.n_items
        n_knapsacks = len(trch_X_train) // n_items
        capacity = self.capacity

        # prepping
        knaps_V_true = [
            get_profits(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)
        ]
        knaps_sol = [
            get_kn_indicators(
                V_true,
                capacity,
                weights=self.weights,
            )
            for V_true in knaps_V_true
        ]
        for k in knaps_sol:
            self.time += k[1]

        if validation:
            n_knapsacks_validation = len(trch_X_validation) // n_items
            knaps_V_true_validation = [
                get_profits(trch_y_validation, kn_nr, n_items)
                for kn_nr in range(n_knapsacks_validation)
            ]
            knaps_sol_validation = [
                get_kn_indicators(
                    V_true,
                    capacity,
                    weights=self.weights,
                )
                for V_true in knaps_V_true_validation
            ]
            for k in knaps_sol_validation:
                self.time += k[1]

        if test:
            n_knapsacks_test = len(trch_X_test) // n_items
            knaps_V_true_test = [
                get_profits(trch_y_test, kn_nr, n_items)
                for kn_nr in range(n_knapsacks_test)
            ]
            knaps_sol_test = [
                get_kn_indicators(
                    V_true,
                    capacity,
                    weights=self.weights,
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
                ### what if for the whole 48 items at a time
                kn_start = kn_nr * n_items
                kn_stop = kn_start + n_items
                train_prediction_error(
                    self.model,
                    optimizer,
                    trch_X_train[kn_start:kn_stop],
                    trch_y_train[kn_start:kn_stop]
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
                            weights=self.weights,
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
                                weights=self.weights,
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
                                weights=self.weights,
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
                            if test:
                                regret_list_test.append(
                                    dict_test["regret_full"]
                                )
                        if self.verbose:
                            if validation:
                                logger.append((dict_epoch, dict_train, dict_validation))
                                print(
                                    "Epoch[{}/{}]::{}, loss(train): {:.6f}, regret(train): {:.2f}, loss(validation): {:.6f}, regret(validation): {:.2f}".format(
                                        epoch + 1,
                                        num_epochs,
                                        cnt,
                                        dict_train["loss"],
                                        dict_train["regret_full"],
                                        dict_validation["loss"],
                                        dict_validation["regret_full"],
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

            if test:
                plt.plot(
                    subepoch_list, regret_list_test
                )
                plt.title("Learning Curve")
                plt.ylabel("Regret")
                plt.xlabel("Sub epoch")
                plt.legend(["SPO-full", "MSE"])
                plt.show()

            elif validation:
                plt.plot(
                    subepoch_list, regret_list, subepoch_list, regret_list_validation
                )
                plt.title("Learning Curve")
                plt.ylabel("Regret")
                plt.xlabel("Sub epoch")
                plt.legend(["training", "validation"])
                plt.show()
            else:
                plt.plot(
                    subepoch_list, regret_list, subepoch_list, regret_list_validation
                )
                plt.title("Learning Curve")
                plt.ylabel("Regret")
                plt.legend(["training", "validation"])
                plt.xlabel("Sub epoch")
                plt.show()

        if self.store_result:
            dd = defaultdict(list)
            for d in test_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            logging.info("Completion Time %s \n" % str(datetime.datetime.now()))
            return df
