import os
from tkinter import X, Y
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

noise_levels = ["0", "0.1", "1.0", "5", "20"]
eval_option = "linear_values" # "reject" or "linear_values"
nr_seeds = 5

title = "Performance of noise levels"
filenames_map = ["mse_learner", "spo_learner_p1", "spo_learner_p1.5", "spo_learner_p2", "spo_learner_p10", "spo_learner_p100", "spo_learner_p1000"]
titles_map = [
    "MSE",
    "SPO P=1",
    "SPO P=1.5",
    "SPO P=2",
    "SPO P=10",
    "SPO P=100",
    "SPO P=1000",
    "SPO Reject"
]
colors_map = [
    "#e41a1c",
    "#377eb8",
    "#000000",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#F781BF"
]

header_names = ["epoch_nr", "validation_regret_full"]

def plot():
    # Set plot vars
    labels = []
    handles = []
    points = []

    for noise_level in noise_levels:
        for file in filenames_map:
            folder = "./results/linear_combination_" + noise_level + "_noise_eval_" + eval_option + "/"
            df_list = []
            for seed_index in range(0, nr_seeds):
                try:
                    df = pd.read_csv(folder + str(seed_index) + '/' + file + ".py", names=header_names, header=None)
                    df_list.append(df)
                except:
                    continue
            regret_list = []

            for df_single in df_list:
                regret_list_validation = df_single.groupby("epoch_nr")["validation_regret_full"].mean()
                regret_list.append(regret_list_validation)

            if len(df_list) == 0:
                continue

            df = pd.concat(regret_list, axis=0)
            final_mean = df.groupby("epoch_nr").mean().iloc[-1]
            filename_without_py = file.split(".py")[0]
            points.append([filename_without_py, noise_level, final_mean])

    for filename in filenames_map:
        line_points = list(filter(lambda x: x[0] == filename, points))
        if len(line_points) < 1:
            continue

        x = [item[1] for item in line_points]
        y = [item[2] for item in line_points]

        name_index = filenames_map.index(filename)
        line_color = colors_map[name_index]
        line_title = titles_map[name_index]

        # Plot legend
        handle, = plt.plot(
            x, y, c=line_color
        )
        handles.append(handle)
        labels.append("validation " + line_title)

    # Plot settings
    plt.title(title + " evaluated by " + eval_option)
    plt.ylabel("Final regret")
    plt.legend(handles=handles, labels=labels)
    plt.xlabel("Noise level")
    plt.show()

plot()
