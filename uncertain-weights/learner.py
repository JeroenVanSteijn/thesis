import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from solving import solveKnapsackProblem, solveKnapsackProblemRelaxation
from sklearn.metrics import confusion_matrix

class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

def get_kn_indicators(
    V_pred, c, weights=None, use_dp=True, relaxation=False, warmstart=None
):
    if weights is None:
        weights = np.ones(V_pred.shape[0])
    if use_dp:
        if relaxation:
            solution = solveKnapsackProblemRelaxation(
                V_pred, weights, c, warmstart=warmstart
            )
        else:
            solution = solveKnapsackProblem(V_pred, weights, c, warmstart=warmstart)
        return np.asarray(solution["assignments"]), solution["runtime"]

    ret = np.zeros(V_pred.shape[0])

    # order by profitability
    V_val = V_pred / weights

    for val in sorted(set(V_val), reverse=True):
        same_val = V_val == val
        tot_weight = sum(weights[same_val])
        if c >= tot_weight:
            # all in
            ret[same_val] = 1
            c = c - tot_weight

        elif c > 0:
            # equal divide
            fraction = c / tot_weight
            ret[same_val] = fraction
            c = 0
            break
        else:
            break
        
    ret[V_pred <= 0] = 0
    return ret

def get_data(trch, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    return trch[kn_start:kn_stop]

def get_profits(trch_y, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    return trch_y[kn_start:kn_stop].data.numpy().T[0]

def get_profits_pred(model, trch_X, kn_nr, n_items):
    kn_start = kn_nr * n_items
    kn_stop = kn_start + n_items
    model.eval()
    with torch.no_grad():
        V_pred = model(Variable(trch_X[kn_start:kn_stop]))
    model.train()
    return V_pred.data.numpy().T[0]

def train_fwdbwd_grad(model, optimizer, sub_X_train, sub_y_train, grad):
    inputs = Variable(sub_X_train, requires_grad=True)
    out = model(inputs)
    grad = grad * torch.ones(1)

    optimizer.zero_grad()

    # backward: hardcode the gradient, let the automatic chain rule backwarding do the rest
    loss = out
    loss.backward(gradient=grad)

    optimizer.step()

def test_fwd(
    model,
    criterion,
    trch_X,
    trch_y,
    n_items,
    capacity,
    knaps_sol,
    weights=None,
    relaxation=False,
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
    regret_smooth = np.zeros(n_knap)
    regret_full = np.zeros(n_knap)
    cf_list = []
    time = 0

    # I should probably just slice the trch_y and preds arrays and feed it like that...
    for kn_nr in range(n_knap):
        V_true = get_profits(trch_y, kn_nr, n_items)
        V_pred = get_profits(V_preds, kn_nr, n_items)
        assignments_pred, t = get_kn_indicators(
            V_pred, c=capacity, weights=weights, relaxation=relaxation
        )

        assignments_true = knaps_sol[kn_nr][0]
        regret_full[kn_nr] = np.sum(V_true * (assignments_true - assignments_pred))
        if not relaxation:
            cf = confusion_matrix(assignments_true, assignments_pred, labels=[0, 1])
            cf_list.append(cf)

        time += t

    info["nonzero_regrsm"] = sum(regret_smooth != 0)
    info["nonzero_regrfl"] = sum(regret_full != 0)
    info["regret_full"] = np.median(regret_full)
    if not relaxation:
        tn, fp, fn, tp = np.sum(np.stack(cf_list), axis=0).ravel()
        info["tn"], info["fp"], info["fn"], info["tp"] = (tn, fp, fn, tp)
        info["accuracy"] = (tn + tp) / (tn + tp + fn + fp)
    else:
        info["accuracy"] = None
    info["runtime"] = time
    return info
