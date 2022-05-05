import csv
import importlib.util
import os
import time
import torch
import numpy
from machineLearner import MLP

def get_filename_for_index(index):
   return str(index)

def load_mod(fileName):
   spec = importlib.util.spec_from_file_location('instance', "./instances/" + fileName + '.py')
   mod = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(mod)
   return mod

def get_features(fileName):
   mod = load_mod(fileName)
   mod.features


def get_actual_processing_times(instance_nr):
   fileName = get_filename_for_index(instance_nr)
   mod = load_mod(fileName)
   return mod.processingTimes

def calculate_cost(schedule, processingTimes):
    cost = 0
    for index, _jobKey in enumerate(schedule):
        waitTime = 0
        for i in range(index):
            waitTime += processingTimes[schedule[i]]
        cost = cost + waitTime
    return cost


def find_optimal_schedule(processingTimes):
    return sorted(processingTimes, key=processingTimes.get)

def tensorToRoundedInt(tensor):
    list = numpy.rint(tensor.cpu().detach().numpy()).tolist()
    result = dict(enumerate(list))
    return result

def run():
    nr_instances = 1
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)
    model.train()

    nr_epochs = 10000
    timestr = time.strftime("%Y%m%d-%H%M%S")

    for i in range(0, nr_instances):
        fileName = get_filename_for_index(i)
        mod = load_mod(fileName)
        actualProcessingTimes = get_actual_processing_times(i)
        dir = 'results/results_' + timestr
        os.makedirs(dir , exist_ok=True)
        with open(dir + '/' + str(i) + '.csv', 'w') as f:
            writer = csv.writer(f)
            def writeXY(x, y):
                writer.writerow([x, y])

            for epoch_nr in range(0, nr_epochs):
                # spo = epoch_nr > nr_epochs / 2 # Do the first half of training through prediction-error based training.
                spo = True

                inputValues = []
                for jobKey in mod.features:
                    inputValues += mod.features[jobKey]
                inputTensor = torch.FloatTensor(inputValues)
                predictions = model(inputTensor)
                print(predictions)

                # SPO+ loss
                if spo:
                    roundedIntegerPredictedValues = tensorToRoundedInt(predictions)
                    predictedSchedule = find_optimal_schedule(roundedIntegerPredictedValues)
                    optimalSchedule = find_optimal_schedule(actualProcessingTimes) # this is theta
                    optimalCost = calculate_cost(optimalSchedule, actualProcessingTimes) # this is v*theta
                    twoPredMinActual = find_optimal_schedule(tensorToRoundedInt(2 * predictions - torch.FloatTensor(list(actualProcessingTimes.values())))) # this is 2theta hat - theta
                    vTwoPredMinActual = calculate_cost(twoPredMinActual, actualProcessingTimes) # this is v*(2theta hat - theta) -> is v* evaluation on the actual processingtimes indeed?
                    deltaL = optimalCost - vTwoPredMinActual

                    optimizer.zero_grad() # -> Tensor' object has no attribute 'zero_grad'
                    predictions.backward(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                new_grad = p.grad * deltaL
                                p.grad.copy_(new_grad)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping

                    optimizer.step()

                    # Write the SPO loss for this epoch
                    predictedScheduleCostOnActual = calculate_cost(predictedSchedule, actualProcessingTimes)
                    loss = predictedScheduleCostOnActual - optimalCost
                    writeXY(epoch_nr, loss)

                # MSE loss
                else:
                    actual = torch.FloatTensor(list(actualProcessingTimes.values()))
                    criterion = torch.nn.MSELoss()
                    loss = criterion(predictions, actual)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



if __name__ == "__main__":
    run()
