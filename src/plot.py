import operator
import os
import pandas as pd
from matplotlib.pyplot import (
    figure,
    plot,
    show,
    title,
    gca
)

def run():
    dirs = os.listdir("results")
    last_dir = max(dirs)
    instances = os.listdir("results/" + last_dir)
    figure()
    for instance in instances:
        file_location = "results/" + last_dir + "/" + instance
        data = pd.read_csv(file_location, header=None)
        groupedByEpoch = data.groupby(data.iloc[:, 0]).mean()
        yData = groupedByEpoch.iloc[:, 1]
        plot(yData, label='Instance ' + instance.replace('.csv', ''))
        
        # Order legend by label sorting
        ax = gca()
        handles, labels = ax.get_legend_handles_labels()
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
        handles2, labels2 = zip(*hl)
        ax.legend(handles2, labels2)

        ax.set_ylabel('SPO loss')
        ax.set_xlabel('Number of epochs')

    
    title("Minimizing wait time, trained on prediction error")
    show()

if __name__ == "__main__":
    run()

    