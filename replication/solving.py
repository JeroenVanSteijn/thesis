from gurobipy import Model, quicksum, GRB

def solveKnapsackGreedily(profits, weights, capacity):
    assert len(profits) == len(weights)
    assert len([x for x in profits if x < 0]) == 0
    assert capacity > 0 and len(profits) >= 2

    n_items = len(profits)
    items = [[profits[i], weights[i], i] for i in range(n_items)]
    items.sort(key=lambda x: float(x[0]) / x[1], reverse=True)
    assert items[0][0] / items[0][1] >= items[1][0] / items[1][1]

    objective = 0
    available_capacity = capacity
    assignments = [
        0 for i in range(n_items)
    ]  # the ith value is 1 if the i-th item is selected
    for i in range(n_items):
        # if the item cannot fit in the remaining capacity
        if items[i][1] > available_capacity:
            continue
        # else, insert it into the knapsack
        objective += items[i][0]  # profit increase
        available_capacity -= items[i][1]  # weight
        assignments[
            items[i][2]
        ] = 1  # note that you selected this item; items[i][2] indicates the original index of the item

        if available_capacity == 0:
            break

    solution_info = {"objective": objective, "assignments": assignments}
    return solution_info


def solveKnapsackProblemRelaxation(
    profits, weights, capacity, warmstart=None, time_limit=None, use_dp=True
):
    profits = [v for v in profits]
    weights = [[w for w in W] for W in weights]
    capacity = [c for c in capacity]
    n = len(profits)

    m = Model()
    m.setParam("OutputFlag", 0)
    x = {}
    for i in range(n):
        x[(i)] = m.addVar(lb=0, ub=1, name="x" + str(i))

    if warmstart is not None:
        for i in range(n):
            x[i].Pstart = warmstart[i]
            m.update()

    m.setObjective(sum((x[i] * profits[i]) for i in range(n)), GRB.MAXIMIZE)
    for w in weights:
        for c in capacity:
            m.addConstr((quicksum(x[i] * w[i] for i in range(n)) <= c))
    m.optimize()

    solution_info = {}
    if m.status == GRB.Status.OPTIMAL:
        solution_info["runtime"] = m.Runtime
        solution_info["objective"] = m.objVal
        m_on = m.getAttr("x", x)
        sol = list(m_on.values())
        solution_info["assignments"] = [i for i in sol]
    else:
        print("SOME EXCEPTION HAPPENED! RETURNING GARBAGE\n")
        val = 0
        solution_info = {}
        solution_info["runtime"] = m.Runtime
        solution_info["objective"] = val
        solution_info["assignments"] = [0 for x in range(len(profits))]
    return solution_info


def solveKnapsackProblem(profits, weights, capacity, warmstart=None):
    profits = [v for v in profits]
    weights = [[w for w in W] for W in weights]
    capacity = [c for c in capacity]

    n = len(profits)
    m = Model()
    m.setParam("OutputFlag", 0)
    x = {}
    for i in range(n):
        x[(i)] = m.addVar(lb=0, ub=1, name="x" + str(i))

    if warmstart is not None:
        for i in range(n):
            x[i].Pstart = warmstart[i]
            m.update()

    m.setObjective(sum((x[i] * profits[i]) for i in range(n)), GRB.MAXIMIZE)
    for w in weights:
        for c in capacity:
            m.addConstr((quicksum(x[i] * w[i] for i in range(n)) <= c))
    m.optimize()

    solution_info = {}
    try:
        if m.status == GRB.Status.OPTIMAL:
            solution_info["runtime"] = m.Runtime
            solution_info["objective"] = m.objVal
            m_on = m.getAttr("x", x)
            sol = list(m_on.values())
            solution_info["assignments"] = [int(i) for i in sol]
    except:
        print("SOME EXCEPTION HAPPENED! RETURNING GARBAGE\n")
        val = 0
        import sys

        solution_info = {}
        solution_info["runtime"] = m.Runtime
        solution_info["objective"] = val
        solution_info["assignments"] = [0 for x in range(len(profits))]
    return solution_info
