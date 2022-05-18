from gurobipy import GRB, Model, quicksum

def solveKnapsackProblem(values, weights, capacity):
    n = len(values)
    m = Model()
    m.setParam('OutputFlag', 0)
    x = {}

    for i in range(n):
        x[i] = m.addVar(lb=0,ub=1, name="x"+str(i)) # Decision variable for 0-1 knapsack
 
    m.setObjective(sum( (x[i]*values[i]) for i in range(n)), GRB.MAXIMIZE)

    m.addConstr((
        quicksum(
            x[i] * weights[i] for i in range(n)
        ) <= capacity
    ))

    m.optimize()       

    solution_info = {}
    try:
        if (m.status == GRB.Status.OPTIMAL):
            solution_info['runtime'] = m.Runtime
            solution_info['objective'] = m.objVal
            m_on = m.getAttr('x',x)
            sol = list(m_on.values())     
            solution_info['assignments'] =  [int(i) for i in sol]
    except:
        print("An exception happend during solving")

    return solution_info

def evaluate_assignment_on_true_values(true_values, assignment):
    result = 0
    for i in assignment:
        result += true_values[i]
    return result