import importlib.util
from milp import FlexibleJobShop

def load_mod(fileName):
   spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
   mod = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(mod)
   return mod
   
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

def get_predicted(instance_nr):
   fileName = 'FJSP_' + str(instance_nr)
   mod = load_mod(fileName)

   predicted_processing_times = mod.processingTimes #TODO: replace this with the predictions based on mod.features

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
   # TODO: can we make cost calculation non-MIP-specific?

def run():
   nr_instances = 5
   for i in range(0, nr_instances):
      predicted_schedule = get_predicted(i)
      optimal_makespan, optimal_schedule = get_optimal(i)
      cost_predicted_schedule = calculate_cost(i, predicted_schedule)

if __name__ == "__main__":
   run()