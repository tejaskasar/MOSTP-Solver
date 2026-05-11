import pandas as pd
import plotly.express as px


# ---------------------------
# Convert solution to table
# ---------------------------
def solution_to_table(solution):
    data = []

    for (i, j, k_, val) in solution:
        data.append({
            "Source": f"S{i+1}",
            "Destination": f"D{j+1}",
            "Mode": f"E{k_+1}",
            "Units": val
        })

    return pd.DataFrame(data)


# ---------------------------
# Compute objective values
# ---------------------------
def compute_objective_values(solution, objective_matrices):
    values = []

    for p in range(len(objective_matrices)):
        total = 0
        for (i, j, k_, val) in solution:
            total += objective_matrices[p][i][j][k_] * val
        values.append(total)

    return values


# ---------------------------
# Plot objective values
# ---------------------------
def plot_objectives(obj_values):
    df = pd.DataFrame({
        "Objective": [f"Obj {i+1}" for i in range(len(obj_values))],
        "Value": obj_values
    })

    fig = px.bar(
        df,
        x="Objective",
        y="Value",
        title="Objective Comparison",
        text_auto=True
    )

    return fig


# ---------------------------
# Plot lambda
# ---------------------------
def plot_lambda(lambda_val):
    df = pd.DataFrame({
        "Metric": ["Lambda"],
        "Value": [lambda_val]
    })

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        title="Compromise Level (λ)",
        text_auto=True
    )

    return fig

def payoff_to_table(payoff):
    df = pd.DataFrame(
        payoff,
        columns=[f"Obj {i+1}" for i in range(len(payoff[0]))]
    )
    
    df.index = [f"Sol {i+1}" for i in range(len(payoff))]
    
    return df

def bounds_to_table(L, U):
    df = pd.DataFrame({
        "Objective": [f"Obj {i+1}" for i in range(len(L))],
        "L (Best)": L,
        "U (Worst)": U
    })
    
    return df