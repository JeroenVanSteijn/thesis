import csv
import random
import numpy as np

nr_items = 11376 # The number of knapsack items to generate
average_weight_value_ratio = 5 # The average in the weight to value ratio (picked from normal distribution)
variance_weight_value_ratio = 0 # The variance in the weight to value ratio (picked from normal distribution)
noiseSize = 0 # amount of noise added to each feature
max_value = 50 # The maximum value for 1 item (picked randomly for interval 0 - max, evenly distributed)

def generate_instances():
    result = []
    for _ in range(0, nr_items):
        value = random.randint(0, max_value)
        weight_value_ratio = np.round(np.random.normal(average_weight_value_ratio, variance_weight_value_ratio))
        weight = np.round(value / weight_value_ratio).astype(int)

        # Generate features that can predict the true weight.
        features = []
        for i in range(0, 10):
            newVal = 0
            if i < 7:
                newVal = np.round(np.random.normal(weight, i) / (i + 1)).astype(int) # TODO: check if/how we like this linear combination.
                newVal = (newVal + random.uniform(-1, 1) * noiseSize).astype(int) 

            else:
                newVal = np.random.normal(weight, i) # TODO: check if/how we like this linear combination.
                newVal = newVal + random.uniform(-1, 1) * noiseSize
            

            features.append(newVal)

        row = features
        row.append(value)
        row.append(weight)

        result.append(row)
    return result

def main():
    f = open('./instances/1.csv', 'w')
    writer = csv.writer(f)
    instances = generate_instances()
    header = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "value", "true_weight"]
    writer.writerow(header)
    for instance in instances:
        writer.writerow(instance)
    f.close()

if __name__ == '__main__':
    main()