import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

noise = "0" # 0 0.1 1.0 5 20"
eval = "rejection" # rejection or linear_values
folder = f"./results/linear_combination_{noise}_noise/"

def find_title_from_folder_name(file_name):
    split = file_name.split("/")
    name = split[-2]
    noise = ''.join(c for c in name if c.isdigit() or c == ".")
    eval = name.split("eval_")[-1]
    eval_nice_name = "with penalty linear in values" if eval == "linear_values" else "with rejection"

    return f"Evaluated {eval_nice_name} on instances with {noise} noise"

title = find_title_from_folder_name(folder)
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
nr_seeds = len(subfolders)

filenames_map = ["mse_learner", "spo_learner_p1_linear_weights", "spo_learner_p1_linear_values", "spo_learner_p2_linear_weights", "spo_learner_p2_linear_values", "spo_learner_p10_linear_weights", "spo_learner_p10_linear_values", "spo_learner_p100_linear_weights", "spo_learner_p100_linear_values", "spo_learner_p1000_linear_weights", "spo_learner_p1000_linear_values", "reject"]
titles_map = [
    "MSE",
    "SPO P=1 linear in weights",
    "SPO P=1 linear in values",
    "SPO P=2 linear in weights",
    "SPO P=2 linear in values",
    "SPO P=10 linear in weights",
    "SPO P=10 linear in values",
    "SPO P=100 linear in weights",
    "SPO P=100 linear in values",
    "SPO P=1000 linear in weights",
    "SPO P=1000 linear in values",
    "SPO Reject"
]
colors_map = [
    "#e41a1c",
    "#377eb8",
    "#377eb8",
    "#4daf4a",
    "#4daf4a",
    "#984ea3",
    "#984ea3",
    "#ff7f00",
    "#ff7f00",
    "#ffff33",
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
        header_names = ["epoch_nr", "validation_regret_full_linear_values", "validation_regret_full_rejection"]
        
        df_list = []
        for seed_index in range(0, nr_seeds):
            nr_files_current_folder = len(os.listdir(folder + str(seed_index) + '/'))
            if nr_files_current_folder == len(files_first_folder):
                df = pd.read_csv(folder + str(seed_index) + '/' + file, names=header_names, header=None)
                df_list.append(df)

        epoch_list = df_list[0]["epoch_nr"].unique()
        regret_list = []

        for df_single in df_list:
            regret_list_validation = df_single.groupby("epoch_nr")["validation_regret_full_" + eval].mean()
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
            epoch_list, means, c=line_color, zorder=name_index
        )
        plt.fill_between(epoch_list, means-stds, means+stds,
            alpha=0.2, edgecolor=line_color, facecolor=line_color, linewidth=0)

        handles.append([handle, name_index])
        labels.append("validation " + line_title)

    # Plot settings
    plt.title(title)

    if eval == "linear_values":
        plt.ylabel("Regret with penalty linear in values (P = 2) for infeasible solutions")
        
    else:
        plt.ylabel("Regret with rejection for infeasible solutions")

    handles, labels = zip(*sorted(zip(handles, labels), key = lambda x,: x[0][1]))
    handles = [handle[0] for handle in handles]

    plt.legend(handles=handles, labels=labels)

    plt.xlabel("Epochs")

    plt.show()

plot()
