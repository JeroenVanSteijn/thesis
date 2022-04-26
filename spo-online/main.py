import csv
import importlib.util
import os
import time
import torch

from milp import FlexibleJobShop
from machineLearner import LinearRegression, MLP

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

def get_predicted_processing_times(instance_nr, model, criterion, optimizer, actualProcessingTimes, epoch, writeXY):
   fileName = get_filename_for_index(instance_nr)
   mod = load_mod(fileName)
   predictedProcessingTimes = {}

   for jobKey in mod.features:
      features = mod.features[jobKey]
      predictedTime = model(torch.FloatTensor(features))
      actualProcessingTime = torch.Tensor([actualProcessingTimes[jobKey]])
      loss = criterion(predictedTime, actualProcessingTime)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      roundedPredictedTime = torch.round(predictedTime)
      predictedProcessingTimes[jobKey] = roundedPredictedTime.item()
    #   writeXY(epoch, loss.item()) => for evaluating the pred error.

   return predictedProcessingTimes

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

def run():
    nr_instances = 5
    model = LinearRegression(5, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()

    nr_epochs = 1000
    timestr = time.strftime("%Y%m%d-%H%M%S")

    for i in range(0, nr_instances):
        dir = 'results/results_' + timestr
        os.makedirs(dir , exist_ok=True)
        with open(dir + '/' + str(i) + '.csv', 'w') as f:
            writer = csv.writer(f)
            def writeXY(x, y):
                writer.writerow([x, y])

            for epoch_nr in range(0, nr_epochs):
                actualProcessingTimes = get_actual_processing_times(i)
                predictedProcessingTimes = get_predicted_processing_times(i, model, criterion, optimizer, actualProcessingTimes, epoch_nr, writeXY)

                # BELOW FOR EXPERIMENT OF SPO LOSS 
                predictedSchedule = find_optimal_schedule(predictedProcessingTimes)
                optimalSchedule = find_optimal_schedule(actualProcessingTimes)

                predictedScheduleCostOnActual = calculate_cost(predictedSchedule, actualProcessingTimes)
                optimalCost = calculate_cost(optimalSchedule, actualProcessingTimes)
                loss = predictedScheduleCostOnActual - optimalCost
                writeXY(epoch_nr, loss)




if __name__ == "__main__":
    run()
