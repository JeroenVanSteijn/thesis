import pandas as pd
import mip

def solve_directly(due_dates, actual_processing_times):
   # Upfront parameters of the problem
   number_jobs = len(due_dates)
   n = range(number_jobs)
   t = range(sum(actual_processing_times))

   # Create model variables
   model = mip.Model()

   # variable: allocation[i][j] is true if at timestep i, job j is running 
   allocation = [[ model.add_var(var_type=mip.BINARY) for _ in t ] for _ in n]

   # constraint: the job should be processed for exactly the same amount of timeslots as the actual processing time.
   for current_job_index in n:
      model += mip.xsum(allocation[i][current_job_index] for i in t) == actual_processing_times[current_job_index]

   # constraint: each timeslot should be selected exactly once.
   for i in t:
      model += mip.xsum(allocation[i][j] for j in n) == 1

   # Solve objective: minimize tardiness
   # f[i] is the finishing time of job i   
   f = [model.add_var(var_type=mip.INTEGER, lb = 1) for _ in n]
   for i in n:
      for j in t:
         # Each timestemp in row i should have a lower or equal value compared to the finishing time.
         model += allocation[j][i] <= f[i] ##TODO: this is incorrect, we need to find the highest index j for the timestep where allocation[j][i] = true...

   # z[i] is the tardiness of job i
   z = [model.add_var(var_type=mip.INTEGER, lb = 0) for _ in n]
   for i in n:
      model += z[i] == f[i] - due_dates[i]

   model.objective = mip.minimize(mip.xsum(z[i] - f[i] for i in n))
   model.optimize()
   print(model.objective_value)

def solve_with_prediction(due_dates, features):
   #TODO: Predict processing times, then create a schedule allocation based on that and the due dates.
   return [3,2,1,0] #This is a very bad example with the heighest optimal cost

def calculate_cost(due_dates, actual_processing_times, allocations):
   total = 0
   current_time = 0
   for current_allocation in allocations:
      current_time += actual_processing_times[current_allocation]
      actual_finishing_time = current_time
      cost = max(due_dates[current_allocation] - actual_finishing_time, 0)
      total += cost
   return total

def run():
   size = 1
   for i in range(size):
      training_set = pd.read_csv("../data/uncertain-processing-time/"+str(i)+".csv")
      features = training_set.iloc[:,0:-1].values
      due_dates = training_set.iloc[:,0].values
      actual_processing_times = training_set.iloc[:,-1].values

      optimal_solution_cost = solve_directly(due_dates, actual_processing_times)
      
      with_prediction_solution = solve_with_prediction(due_dates, features)
      with_prediction_cost = calculate_cost(due_dates, actual_processing_times, with_prediction_solution)

      # decision_error_based_on_predictions = with_prediction_cost - optimal_solution_cost

if __name__ == "__main__":
   run()