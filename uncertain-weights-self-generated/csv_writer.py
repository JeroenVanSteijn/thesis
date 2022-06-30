import csv
import os

def write_results(file_name, info):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    f = open(file_name, 'w+')
    writer = csv.writer(f)
    # header = ["epoch_nr", "subepoch", "train_regret_full", "validation_regret_full"]
    for line in info:
        writer.writerow([line["epoch_nr"], line["subepoch"], line["train_regret_full"],line["validation_regret_full"]])
    f.close()