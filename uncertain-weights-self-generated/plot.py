import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

title = "Reject in training"
folder = "./results/linear_combination_20_noise/"
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
nr_seeds = len(subfolders)

filenames_map = ["mse_learner", "spo_learner_p1", "spo_learner_p2", "spo_learner_p10", "spo_learner_p100", "spo_learner_p1000", "reject"]
titles_map = [
    "MSE",
    "SPO P=1",
    "SPO P=2",
    "SPO P=10",
    "SPO P=100",
    "SPO P=1000",
    "SPO Reject"
]
colors_map = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#F781BF"
]

def plot():
    files_first_folder = os.listdir(subfolders[0])

    # Set plot vars
    labels = []
    handles = []
    
    # Iterate over the files in the first folder.
    for file in files_first_folder:
        header_names = ["epoch_nr", "validation_regret_full"]
        
        df_list = []
        for seed_index in range(0, nr_seeds):
            nr_files_current_folder = len(os.listdir(folder + str(seed_index) + '/'))
            if nr_files_current_folder == len(files_first_folder):
                df = pd.read_csv(folder + str(seed_index) + '/' + file, names=header_names, header=None)
                df_list.append(df)

        epoch_list = df_list[0]["epoch_nr"].unique()
        regret_list = []

        for df_single in df_list:
            regret_list_validation = df_single.groupby("epoch_nr")["validation_regret_full"].mean()
            regret_list.append(regret_list_validation)

        df = pd.concat(regret_list, axis=0)

        means = df.groupby("epoch_nr").mean()
        stds = df.groupby("epoch_nr").sem()
        
        filename_without_py = file.split(".py")[0]
        name_index = filenames_map.index(filename_without_py)
        line_color = colors_map[name_index] 
        line_title = titles_map[name_index]

        # Plot legend
        handle, = plt.plot(
            epoch_list, means, c=line_color
        )
        plt.fill_between(epoch_list, means-stds, means+stds,
            alpha=0.2, edgecolor=line_color, facecolor=line_color, linewidth=0)

        handles.append(handle)
        labels.append("validation " + line_title)

    # Plot settings
    plt.title(title)
    plt.ylabel("Regret")

    handles, labels = zip(*sorted(zip(handles, labels), key = lambda x,: x[1]))

    plt.legend(handles=handles, labels=labels)

    plt.xlabel("Epochs")

    plt.show()

plot()