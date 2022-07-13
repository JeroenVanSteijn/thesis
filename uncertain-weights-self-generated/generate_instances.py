import csv
import random
import numpy as np

nr_items = 4000 # The number of knapsack items to generate
average_weight_value_ratio = 5 # The average in the weight to value ratio (picked from normal distribution)
variance_weight_value_ratio = 0 # The variance in the weight to value ratio (picked from normal distribution)
noiseSize = 0.001 # amount of noise added to each feature
max_value = 50 # The maximum value for 1 item (picked randomly for interval 0 - max, evenly distributed)
random_features = True
filename = "multi_realization.csv"

generate_multiple_realizations_small_sample = True # Experiment idea from example by Mathijs.

def generate_instances():
    result = []
    for _ in range(0, nr_items):
        value = random.randint(0, max_value)
        weight_value_ratio = min(0.1, np.random.normal(average_weight_value_ratio, variance_weight_value_ratio))
        weight = np.round(value / weight_value_ratio).astype(int)

        # Generate features that can predict the true weight.
        features = []
        for i in range(0, 9):
            newVal = 0
            if i < 6:
                if random_features:
                    newVal = random.randint(0, 100)
                else:
                    newVal = np.round(np.random.normal(weight, i) / (i + 1)).astype(int) # TODO: check if/how we like this linear combination.
                    newVal = (newVal + random.uniform(-1, 1) * noiseSize).astype(int) 

            else:
                if random_features:
                    newVal = random.random()
                else:
                    newVal = np.random.normal(weight, i) # TODO: check if/how we like this linear combination.
                    newVal = newVal + random.uniform(-1, 1) * noiseSize
            

            features.append(newVal)

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
    f = open('./instances/' + filename, 'w')
    writer = csv.writer(f)
    if generate_multiple_realizations_small_sample:
        instances = generate_instances_multi_realize()
    else:
        instances = generate_instances()
    header = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "value", "true_weight"]
    writer.writerow(header)
    for instance in instances:
        writer.writerow(instance)
    f.close()

if __name__ == '__main__':
    main()