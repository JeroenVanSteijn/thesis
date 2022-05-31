import csv
import os
import time
import torch
import numpy
from machineLearner import LinearRegression
from knapSack import solveKnapsackProblem, evaluate_assignment_on_true_values

def map_array_of_objects_to_key(array_of_objects, key):
    result = []
    for item in array_of_objects:
        result.append(item[key])
    return result

def tensorToRoundedInt(tensor):
    list = numpy.rint(tensor.cpu().detach().numpy()).tolist()
    result = dict(enumerate(list))
    return result

instances = [[
    {
        "weigth": 2,
        "value": 3,
        "features": [0,0,0,0,0]
    },
    {
        "weigth": 4,
        "value": 5,
        "features": [1,1,1,1,1]
    },
    {
        "weigth": 6,
        "value": 7,
        "features": [2,2,2,2,2]
    },
    {
        "weigth": 8,
        "value": 12,
        "features": [3,3,3,3,3]
    },
    {
        "weigth": 2,
        "value": 15,
        "features": [4,4,4,4,4]
    }
]]

capacity = 10

def run():
    nr_instances = len(instances)
    model = LinearRegression(25, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)
    model.train()

    nr_epochs = 1000
    timestr = time.strftime("%Y%m%d-%H%M%S")

    for i in range(0, nr_instances):
        instance = instances[i]
        trueValues = map_array_of_objects_to_key(instance, "value")
        weights = map_array_of_objects_to_key(instance, "weigth")
        optimalCost = solveKnapsackProblem(trueValues, weights, capacity)["objective"] # this is v*theta, i.e. the cost of the optimal schedule.

        dir = 'results/results_' + timestr
        os.makedirs(dir , exist_ok=True)
        with open(dir + '/' + str(i) + '.csv', 'w') as f:
            writer = csv.writer(f)
            def writeXY(x, y):
                writer.writerow([x, y])

            for epoch_nr in range(0, nr_epochs):

                # WARM STARTING:
                # spo = epoch_nr > nr_epochs / 2 # Do the first half of training through prediction-error based training.
                spo = True # Disables warm starting

                inputValues = []
                for item in instance:
                    inputValues += item['features'] # concat all feature arrays in-order

                # FIXME: we make all predictions at once for now! This gives a dependency on order and on the amount of predictions to be made.
                inputTensor = torch.FloatTensor(inputValues)
                predictions = model(inputTensor) # The estimated values array
                predictedValues = tensorToRoundedInt(predictions)

                # SPO+ loss
                if spo:
                    solvedForPredicted = solveKnapsackProblem(predictedValues, weights, capacity)
                    
                    twoPredMinActual = []
                    for index, predictedValue in enumerate(predictedValues):
                        twoPredMinActual.append(2 * predictedValue - instance[index]["value"])

                    vTwoPredMinActual = solveKnapsackProblem(twoPredMinActual, weights, capacity)["objective"]
                    deltaL = optimalCost - vTwoPredMinActual

                    optimizer.zero_grad()

                    grad = deltaL * torch.ones(5)

                    predictions.backward(gradient=grad)
                    optimizer.step()

                    # Write the SPO loss for this epoch
                    predictedScheduleCostOnActual = evaluate_assignment_on_true_values(trueValues, solvedForPredicted["assignments"])

                    # We're maximizing
                    loss = optimalCost - predictedScheduleCostOnActual 
                    writeXY(epoch_nr, loss)

                # MSE loss
                else:
                    actual = torch.FloatTensor(trueValues)
                    criterion = torch.nn.MSELoss()
                    loss = criterion(predictions, actual)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

if __name__ == "__main__":
    run()
