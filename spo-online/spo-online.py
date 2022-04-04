import pandas as pd
import mip

def solve_directly(number_machines, dependencies, processing_times):
   # Upfront parameters of the problem
   number_jobs = len(processing_times)

   # Create model variables
   model = mip.Model()
   c = model.add_var(name="C")

   # TODO: Create MIP for multi-machine problem.
   model.objective = c
   model.optimize()
   print(model.objective_value)

def solve_with_prediction(dependencies, features):
   predicted_processing_times = predict_processing_times(features)
   # TODO: calculate the cost of the allocation with the actual processing times.
   return solve_directly(dependencies, predicted_processing_times)

def train_with_prediction(dependencies, features):
   predicted_processing_times = predict_processing_times(features)
   solve_directly(dependencies, predicted_processing_times)
   # TODO: calculate the cost of the allocation with the actual processing times.
   # TODO: return the decision error to the predictor here.

def predict_processing_times(features):
   return [1, 1] # TODO  

def run():
   size = 1
   training = True
   for i in range(size):
      number_machines = pd.read_csv("../data/uncertain-processing-time/"+str(i)+"/nr_machines.csv")[0][0]
      jobs = pd.read_csv("../data/uncertain-processing-time/"+str(i)+"/job_features.csv")
      dependencies = pd.read_csv("../data/uncertain-processing-time/"+str(i)+"/dependencies.csv")
      features = jobs.iloc[:,0:-1].values
      actual_processing_times = jobs.iloc[:,-1].values

      optimal_solution_cost = solve_directly(number_machines, dependencies, actual_processing_times)

      if training:
         train_with_prediction(dependencies, features)
      else:
         with_prediction_cost = solve_with_prediction(dependencies, features)
         decision_error_based_on_predictions = with_prediction_cost - optimal_solution_cost
         print(decision_error_based_on_predictions)

if __name__ == "__main__":
   run()