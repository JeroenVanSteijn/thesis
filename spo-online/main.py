import importlib.util
import torch

from milp import FlexibleJobShop
from machineLearner import LinearRegression

def load_mod(fileName):
   spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
   mod = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(mod)
   return mod

def get_features(fileName):
   mod = load_mod(fileName)
   mod.features

def get_optimal(instance_nr):
   fileName = 'FJSP_' + str(instance_nr)
   mod = load_mod(fileName)

   mip = FlexibleJobShop(
      jobs=mod.jobs, 
      machines=mod.machines, 
      processingTimes=mod.processingTimes, 
      machineAlternatives=mod.machineAlternatives, 
      operations=mod.operations,
      instance=fileName,
      changeOvers=mod.changeOvers,
      orders=mod.orders
   )
   optimal_makespan, optimal_schedule = mip.build_model()

   return optimal_makespan, optimal_schedule

def get_predicted_processing_times(instance_nr, model, criterion, optimizer, isTraining):
   fileName = 'FJSP_' + str(instance_nr)
   mod = load_mod(fileName)
   predictedProcessingTimes = {}
   actualProcessingTimes = mod.processingTimes
   
   # Note that we do these epochs from the start as a sort of warm start...
   nr_epochs = 100
   for _ in range(nr_epochs):
      for jobKey in mod.features:
         features = mod.features[jobKey]
         predictedTime = model(torch.FloatTensor(features))
         roundedPredictedTime = int(torch.round(predictedTime).item())
         actualProcessingTime = torch.Tensor([actualProcessingTimes[jobKey]])
         predictedProcessingTimes[jobKey] = roundedPredictedTime

         if isTraining:
            loss = criterion(predictedTime, actualProcessingTime)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

   return predictedProcessingTimes

def get_predicted_schedule(instance_nr, predicted_processing_times):
   fileName = 'FJSP_' + str(instance_nr)
   mod = load_mod(fileName)

   mip = FlexibleJobShop(
      jobs=mod.jobs, 
      machines=mod.machines, 
      processingTimes=predicted_processing_times, 
      machineAlternatives=mod.machineAlternatives, 
      operations=mod.operations,
      instance=fileName,
      changeOvers=mod.changeOvers,
      orders=mod.orders
   )
   _, opt_schedule_predicted = mip.build_model()

   return opt_schedule_predicted

def calculate_cost(instance_nr, schedule):
   fileName = 'FJSP_' + str(instance_nr)
   # TODO: Calculate cost of the schedule which is based on the predicted processing times when working with the actual processing times.
   return 5

def run():
   nr_instances = 5
   model = LinearRegression(5, 1)
   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
   model.train()

   for i in range(0, nr_instances):
      predicted_processing_times = get_predicted_processing_times(i, model, criterion, optimizer, True)
      predicted_schedule = get_predicted_schedule(i, predicted_processing_times)
      optimal_makespan, optimal_schedule = get_optimal(i)
      cost_predicted_schedule = calculate_cost(i, predicted_schedule)

if __name__ == "__main__":
   run()