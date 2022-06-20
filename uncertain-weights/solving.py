from gurobipy import Model, quicksum, GRB

def solveKnapsackProblem(weights, profits, capacity, warmstart=None):
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
        solution_info = {}
        solution_info["runtime"] = m.Runtime
        solution_info["objective"] = val
        solution_info["assignments"] = [0 for x in range(len(profits))]
    return solution_info
