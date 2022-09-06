Main challenge with uncertainty in the constraints is how to handle infeasible solutions.

An initial response to this might be to evaluate infeasible solutions as having 0 value (throwing all of the knapsack items out), but this causes a loss of information that can prevent convergence to an optimal result.

Another solution taken in other fields as well is to penalize infeasible solutions for each item that is over-weight, linearly in their value. This penalization can happen linearly in the weight or value of the items.

The question then arizes how much the penalization should be. Penalizing under 1 * the value of the items causes infeasible solutions to become preferable, which is not desirable. Thus the penalty should be at least 1 * the value of the item.

With exactly 1 * the value of the item, it is not better, nor worse, to take an overweight items. This means no preference is created towards feasible solutions, which seems desirable in almost all cases.

Finally, with higher penalization, it can be hypothesized that a more conservative prediction will be made, since it makes it very undesirable to have overweight items. This conservatism will only last untill completely feasible solutions are found.

Now, in order to compare the regret of these different options (training using rejection, repairation (P = 1), P > 1 linear in weight/value), they should be evaluated on the same regret calculation. This regret calculation should give insight into:
1. The amount of feasible solutions
2. How feasible/infeasible they were
3. If all solutions were feasible, how good they were compared to optimal ('normal' regret)

To this end, two evaluation approaches are selected for the regret:
1. Rejection
2. Evaluation

The calculation goes as follows:
- Calculating the regret on all items that fit in the knapsack
- If any items are found that do not fit in the knapsack, for rejection, evaluate as 0 objective value. For penalization, apply a penalty linear in the value of the over-weight items.

