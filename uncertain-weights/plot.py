import os
import pandas as pd
from matplotlib import pyplot as plt

title = "Penalty function linear in values on instances generated with 10 noise"
folder = "./results/28-06_10_noise_multi_p/"
use_subepoch = False # If true we plot epochs (less detailed)

def plot():
    files = os.listdir(folder)
    legend = []
    for file in files:
        header_names = ["epoch_nr", "subepoch", "train_regret_full", "validation_regret_full"]
        df = pd.read_csv(folder + file, names=header_names, header=None)

        if use_subepoch:
            subepoch_list = df["subepoch"]
            regret_list = df["train_regret_full"]
            regret_list_validation = df["validation_regret_full"]
            plt.plot(
                subepoch_list, regret_list, subepoch_list, regret_list_validation
            )
        else:
            epoch_list = df["epoch_nr"].unique()
            regret_list = df.groupby("epoch_nr")["train_regret_full"].mean()
            regret_list_validation = df.groupby("epoch_nr")["validation_regret_full"].mean()
            plt.plot(
                epoch_list, regret_list, epoch_list, regret_list_validation
            )


        legend.append("training " + file.split(".py")[0])
        legend.append("validation " + file.split(".py")[0])


    plt.title(title)
    plt.ylabel("Regret")
    plt.legend(legend)
    plt.xlabel("Sub Epochs" if use_subepoch else "Epochs")

    plt.show()

plot()