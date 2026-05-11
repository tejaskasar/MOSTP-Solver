from pulp import *

def solve_single_objective(cost_matrix, supply, demand, capacity):
    """
    Solve MSTP for a single objective
    cost_matrix: 3D list [i][j][k]
    """

    m = len(supply)
    n = len(demand)
    k = len(capacity)

    # Create model
    model = LpProblem("MSTP_Single_Objective", LpMinimize)

    # Decision variables x[i][j][k]
    x = LpVariable.dicts("x", (range(m), range(n), range(k)), lowBound=0)

    # ---------------------------
    # Objective Function
    # ---------------------------
    model += lpSum(
        cost_matrix[i][j][k_] * x[i][j][k_]
        for i in range(m)
        for j in range(n)
        for k_ in range(k)
    )

    # ---------------------------
    # Constraints
    # ---------------------------

    # Supply constraints
    for i in range(m):
        model += lpSum(x[i][j][k_] for j in range(n) for k_ in range(k)) == supply[i]

    # Demand constraints
    for j in range(n):
        model += lpSum(x[i][j][k_] for i in range(m) for k_ in range(k)) == demand[j]

    # Capacity constraints
    for k_ in range(k):
        model += lpSum(x[i][j][k_] for i in range(m) for j in range(n)) == capacity[k_]

    # ---------------------------
    # Solve
    # ---------------------------
    model.solve()

    # ---------------------------
    # Extract Solution
    # ---------------------------
    solution = []

    for i in range(m):
        for j in range(n):
            for k_ in range(k):
                val = x[i][j][k_].value()
                if val is not None and val > 0:
                    solution.append((i, j, k_, val))

    # Objective value
    obj_value = value(model.objective)

    return solution, obj_value




def compute_L_U(objective_matrices, supply, demand, capacity):
    """
    Compute L and U for all objectives
    """

    num_obj = len(objective_matrices)

    solutions = []
    values = []

    # ---------------------------
    # Step 1: Solve each objective
    # ---------------------------
    for p in range(num_obj):
        sol, val = solve_single_objective(
            objective_matrices[p], supply, demand, capacity
        )
        solutions.append(sol)

    # ---------------------------
    # Step 2: Build Payoff Matrix
    # ---------------------------
    for sol in solutions:
        row = []

        for p in range(num_obj):
            total = 0

            for (i, j, k_, val) in sol:
                total += objective_matrices[p][i][j][k_] * val

            row.append(total)

        values.append(row)

    # ---------------------------
    # Step 3: Compute L and U
    # ---------------------------
    L = []
    U = []

    for p in range(num_obj):
        # Lower bound (diagonal)
        L.append(values[p][p])

        # Upper bound (max in column)
        col_vals = [values[row][p] for row in range(num_obj)]
        U.append(max(col_vals))

    return L, U, values


def solve_fuzzy(objective_matrices, supply, demand, capacity, L, U):
    """
    Solve final fuzzy optimization problem
    """

    m = len(supply)
    n = len(demand)
    k = len(capacity)
    p = len(objective_matrices)

    # Create model
    model = LpProblem("Fuzzy_MSTP", LpMaximize)

    # Decision variables
    x = LpVariable.dicts("x", (range(m), range(n), range(k)), lowBound=0)

    # Lambda variable
    lambda_var = LpVariable("lambda", lowBound=0)

    # ---------------------------
    # Build objective functions
    # ---------------------------
    F = []

    for obj in range(p):
        expr = lpSum(
            objective_matrices[obj][i][j][k_] * x[i][j][k_]
            for i in range(m)
            for j in range(n)
            for k_ in range(k)
        )
        F.append(expr)

    # ---------------------------
    # Add constraints
    # ---------------------------

    # Supply
    for i in range(m):
        model += lpSum(x[i][j][k_] for j in range(n) for k_ in range(k)) == supply[i]

    # Demand
    for j in range(n):
        model += lpSum(x[i][j][k_] for i in range(m) for k_ in range(k)) == demand[j]

    # Capacity
    for k_ in range(k):
        model += lpSum(x[i][j][k_] for i in range(m) for j in range(n)) == capacity[k_]

    # ---------------------------
    # Fuzzy constraints (CORE)
    # ---------------------------
    for obj in range(p):
        if U[obj] != L[obj]:  # avoid division issue
            model += F[obj] + (U[obj] - L[obj]) * lambda_var <= U[obj]
        else:
            model += F[obj] <= U[obj]

    # ---------------------------
    # Objective: Maximize λ
    # ---------------------------
    model += lambda_var

    # ---------------------------
    # Solve
    # ---------------------------
    model.solve()

    # ---------------------------
    # Extract solution
    # ---------------------------
    solution = []

    for i in range(m):
        for j in range(n):
            for k_ in range(k):
                val = x[i][j][k_].value()
                if val is not None and val > 0:
                    solution.append((i, j, k_, val))

    lambda_value = lambda_var.value()

    return solution, lambda_value


def perturb_parameter(values, index, delta):
    """
    Change ONE parameter by delta
    """

    new_values = values.copy()

    new_values[index] += delta

    return new_values
