import csv
import os

def write_results(file_name, info):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    f = open(file_name, 'w+')
    writer = csv.writer(f)
    for line in info:
        writer.writerow([line["epoch_nr"], line["validation_regret_full"]])
    f.close()