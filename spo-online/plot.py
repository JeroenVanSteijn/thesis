import os
import pandas as pd
from matplotlib.pyplot import (
    axhline,
    clf,
    figure,
    gca,
    legend,
    plot,
    fill_between,
    show,
    title,
    xlabel,
    ylabel,
)

def run():
    dirs = os.listdir("results")
    figureIndex = 0
    last_dir = max(dirs)
    instances = os.listdir("results/" + last_dir)
    for instance in instances:
        file_location = "results/" + last_dir + "/" + instance
        data = pd.read_csv(file_location)
        groupedByEpoch = data.groupby(data.iloc[:, 0]).mean()
        yData = groupedByEpoch.iloc[:, 1]
        figure()
        figureIndex = figureIndex + 1
        plot(yData, label=instance)
    
    show()
    title("Experiment " + last_dir)

if __name__ == "__main__":
    run()

    