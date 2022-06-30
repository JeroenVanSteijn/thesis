import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

title = "Penalty function linear in values on instances generated with random features"
folder = "./results/30-06_random_noise_multi_p/"
use_subepoch = False # If true we plot epochs (less detailed)
plot_training = False # If true also plot the training curve

def plot():
    files = os.listdir(folder)
    labels = []
    handles = []
    n = len(files) * 2 if plot_training else len(files)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
    for file in files:
        header_names = ["epoch_nr", "subepoch", "train_regret_full", "validation_regret_full"]
        df = pd.read_csv(folder + file, names=header_names, header=None)

        if use_subepoch:
            subepoch_list = df["subepoch"]
            regret_list = df["train_regret_full"]
            regret_list_validation = df["validation_regret_full"]
            if plot_training:
                handle, = plt.plot(
                    subepoch_list, regret_list, c=next(color)
                )
                handles.append(handle)
            handle, = plt.plot(
                subepoch_list, regret_list_validation, c=next(color)
            )
            handles.append(handle)

        else:
            epoch_list = df["epoch_nr"].unique()
            regret_list = df.groupby("epoch_nr")["train_regret_full"].mean()
            regret_list_validation = df.groupby("epoch_nr")["validation_regret_full"].mean()
            if plot_training:
                handle, = plt.plot(
                    subepoch_list, regret_list, c=next(color)
                )
                handles.append(handle)
            handle, = plt.plot(
                epoch_list, regret_list_validation, c=next(color)
            )
            handles.append(handle)

        if plot_training:
            labels.append("training " + file.split(".py")[0])
        labels.append("validation " + file.split(".py")[0])

    plt.title(title)
    plt.ylabel("Regret")

    handles, labels = zip(*sorted(zip(handles, labels), key = lambda x,: x[1]))

    plt.legend(handles=handles, labels=labels)

    plt.xlabel("Sub Epochs" if use_subepoch else "Epochs")

    plt.show()

plot()