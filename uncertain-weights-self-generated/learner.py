import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from solving import solveKnapsackProblem

class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

def get_kn_indicators(
    V_pred, c, values, warmstart=None
):
    solution = solveKnapsackProblem([V_pred.round().astype(int).tolist()], values, c, warmstart=warmstart)
    assignments = np.asarray(solution["assignments"])
    return assignments, solution["runtime"]

def get_objective_value_penalized_infeasibility(assignments, true_weights, values, capacity, penalty_P, penalty_function_type, predictions):
    # First order the selected items by weight to value ratio
    selectedItems = []
    for index, element in enumerate(assignments):
        if element == 1:
            value = values[index]
            ratio = value / predictions[index]
            selectedItems.append([index, value, ratio])

    def getSortKey(elem):
        return elem[2] #return the ratio.

    # sort descending
    selectedItems.sort(key=getSortKey, reverse=True)

    # Start the calculation
    capacity = capacity[0]
    capacity_used = 0
    total_value = 0
    infeasible = False

    for element in selectedItems:
        index = element[0]
        value = element[1]

        total_value += values[index]

        new_total_capacity_used = capacity_used + true_weights[index]
        if new_total_capacity_used <= capacity:
            capacity_used = new_total_capacity_used
        else:
            infeasible = True
            if penalty_function_type == "linear_weights":
                total_value -= true_weights[index] * penalty_P

            elif penalty_function_type == "linear_values":
                total_value -= values[index] * penalty_P

            elif penalty_function_type == "repair":
                total_value -= values[index]

            elif penalty_function_type == "reject":
                return 0, True

            else:
                raise Exception("Invalid penalty function type.")
    
    return total_value, infeasible

def get_data(trch, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    return trch[kn_start:kn_stop]

def get_weights(trch_y, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    return trch_y[kn_start:kn_stop].data.numpy().T[0]

def get_values(values, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    return values[kn_start:kn_stop]

def get_weights_pred(model, trch_X, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    model.eval()
    with torch.no_grad():
        V_pred = model(Variable(trch_X[kn_start:kn_stop]))
    model.train()
    return V_pred.data.numpy().T[0]

# Test fwdbwd mse is a training mechanism. It trains the model based on the built in gradient that comes from calculating the MSE loss.
def train_fwdbwd_mse(model, optimizer, sub_X_train, y_true):
    inputs = Variable(sub_X_train, requires_grad=True)
    out = model(inputs)

    optimizer.zero_grad()
    
    criterion = torch.nn.MSELoss()
    loss = criterion(out, y_true)
    loss.backward()

    optimizer.step()

# Test fwdbwd grad is a training mechanism. It trains the model based on an earlier calculed gradient.
def train_fwdbwd_grad(model, optimizer, sub_X_train, sub_y_train, grad):
    inputs = Variable(sub_X_train, requires_grad=True)
    out = model(inputs)
    grad = grad * torch.ones(1)

    optimizer.zero_grad()

    # backward: hardcode the gradient, let the automatic chain rule backwarding do the rest
    loss = out
    loss.backward(gradient=grad)

    optimizer.step()

# Test fwd is an evaluation mechanism. It executes the model without training.
def test_fwd(
    model,
    criterion,
    trch_X,
    trch_y,
    n_items,
    capacity,
    knaps_sol,
    values
):
    info = dict()
    model.eval()
    with torch.no_grad():
        # compute loss on whole dataset
        inputs = Variable(trch_X)
        target = Variable(trch_y)
        V_preds = model(inputs)
        info["loss"] = criterion(V_preds, target).item()
    model.train()

    n_knap = len(V_preds) // n_items
    regret_full_linear_values = np.zeros(n_knap)
    regret_full_rejection = np.zeros(n_knap)
    time = 0

    # I should probably just slice the trch_y and preds arrays and feed it like that...
    for kn_nr in range(n_knap):
        V_true = get_weights(trch_y, kn_nr, n_items)
        values_specific = get_values(values, kn_nr, n_items)
        V_pred = get_weights(V_preds, kn_nr, n_items)

        assignments_pred, t = get_kn_indicators(
            V_pred, c=capacity, values=values_specific
        )

        assignments_true = knaps_sol[kn_nr][0]

        optimal_value = np.sum(values_specific * (assignments_true))

        # Calculate values either with rejection or linear_values P = 2 for evaluation.
        achieved_value_linear, _was_penalized = get_objective_value_penalized_infeasibility(
            assignments_pred, V_true, values_specific, capacity, 2, "linear_values", V_pred
        )
        regret_full_linear_values[kn_nr] = optimal_value - achieved_value_linear
        
        achieved_value_rejection, _was_penalized_rejection =get_objective_value_penalized_infeasibility(
            assignments_pred, V_true, values_specific, capacity, 0, "reject", V_pred
        )
        regret_full_rejection[kn_nr] = optimal_value - achieved_value_rejection


        time += t

    info["regret_full_linear_values"] = np.median(regret_full_linear_values)
    info["regret_full_rejection"] = np.median(regret_full_rejection)
    info["runtime"] = time

    return info
