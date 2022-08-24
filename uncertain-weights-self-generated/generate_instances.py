import os
import csv
import random
import numpy as np

nr_seeds = 5 # The number of times to repeat the procedure on different seeds.
nr_items = 11376 # The number of knapsack items to generate
average_weight_value_ratio = 0.2 # The average in the weight to value ratio (picked from normal distribution)
variance_weight_value_ratio = 0 # The variance in the weight to value ratio (picked from normal distribution)
noiseSize = 20 # amount of noise added to each feature
random_features = True
foldername = "linear_combination_20_noise"

generate_multiple_realizations_small_sample = False # Experiment idea from example by Mathijs.
# Otherwise: Kim's suggestion of linear combination.

def generate_instances_linear_combination():
    linear_c = []
    for i in range (0, 9):
        linear_c.append(random.uniform(0.2, 1))

    result = []
    for _ in range(0, nr_items):
        weight_value_ratio = max(0.1, np.random.normal(average_weight_value_ratio, variance_weight_value_ratio))

        # Generate features that can predict the true weight.
        features = []
        for i in range(0, 9):
            newVal = random.uniform(1, 10)
            features.append(newVal)

        weight = np.round(sum([linear_c[i] * features[i] for i in range(0, 9)]) + random.uniform(-noiseSize, noiseSize)).astype(int) # somewhere between 1 and 90
        value = np.round(weight / weight_value_ratio).astype(int)

        row = features
        row.append(value)
        row.append(weight)

        result.append(row)
    return result

def generate_instances_multi_realize():
    item1_weights = [11,12,13,14,15,16,17,18,19,20]
    item1 = [1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 15]
    item2_weights = [1,2,3,4,5,6,7,8,9,10]
    item2 = [2, 2, 2, 2, 2, 2, 2.0, 2.0, 2.0, 5]
    item3_weights = [6,7,8,9,10,11,12,13,14,15]
    item3 = [3, 3, 3, 3, 3, 3, 3.0, 3.0, 3.0, 10]

    result = []

    for _ in range(0, nr_items):
        i1 = [*item1, *[random.choice(item1_weights)]]
        result.append(i1)
        i2 = [*item2, *[random.choice(item2_weights)]]
        result.append(i2)
        i3 = [*item3, *[random.choice(item3_weights)]]
        result.append(i3)

    return result


def main():
    for index in range(0, nr_seeds):
        filename = './instances/' + foldername + '/' + str(index) + '.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, 'w')
        writer = csv.writer(f)
        if generate_multiple_realizations_small_sample:
            instances = generate_instances_multi_realize()
        else:
            instances = generate_instances_linear_combination()
        header = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "value", "true_weight"]
        writer.writerow(header)
        for instance in instances:
            writer.writerow(instance)
        f.close()

if __name__ == '__main__':
    main()