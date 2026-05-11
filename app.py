import streamlit as st
import pandas as pd
from utils import convert_to_3d
from solver import solve_single_objective, compute_L_U, solve_fuzzy, perturb_parameter
from visualization import *
st.set_page_config(page_title="MSTP Solver", layout="wide")

st.title(" Multiobjective Solid Transportation Problem Solver")

# ---------------------------
# STEP 1: Problem Dimensions
# ---------------------------
st.header(" Problem Setup")

col1, col2, col3, col4 = st.columns(4)

with col1:
    m = st.number_input("Sources", 1, 10, 2)

with col2:
    n = st.number_input("Destinations", 1, 10, 3)

with col3:
    k = st.number_input("Modes", 1, 5, 2)

with col4:
    p = st.number_input("Objectives", 1, 10, 2)

# ---------------------------
# STEP 2: Supply, Demand, Capacity
# ---------------------------
st.header(" Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Supply")
    supply = [st.number_input(f"S{i+1}", min_value=0.0, value=10.0) for i in range(m)]

with col2:
    st.subheader("Demand")
    demand = [st.number_input(f"D{j+1}", min_value=0.0, value=10.0) for j in range(n)]

with col3:
    st.subheader("Capacity")
    capacity = [st.number_input(f"E{i+1}", min_value=0.0, value=10.0) for i in range(k)]
# ---------------------------
# STEP 3: Objective Matrices
# ---------------------------
st.header(" Objective Functions")

objectives = []

for obj in range(p):
    st.subheader(f"Objective {obj+1}")
    
    # Create empty table
    cols = []
    for j in range(n):
        for k_ in range(k):
            cols.append(f"D{j+1}-E{k_+1}")
    
    df = pd.DataFrame(
        [[0.0]*len(cols) for _ in range(m)],
        columns=cols,
        index=[f"S{i+1}" for i in range(m)]
    )
    
    edited_df = st.data_editor(df, key=f"obj_{obj}")
    
    objectives.append(edited_df)

# ---------------------------
# STEP 4: Validate Input
# ---------------------------
st.header(" Validation")

if sum(supply) != sum(demand):
    st.error("❌ Supply must equal Demand")
elif sum(supply) != sum(capacity):
    st.warning("⚠️ Supply ≠ Capacity (May be infeasible)")
else:
    st.success("✔ Balanced Problem")

# ---------------------------
# Ordinary Sensitivity Analysis
# ---------------------------

st.header(" Ordinary Sensitivity Analysis")

sens_category = st.selectbox(
    "Select Parameter Type",
    ["Supply", "Demand", "Capacity"]
)

if sens_category == "Supply":
    selected_index = st.selectbox(
        "Select Supply Variable",
        [f"S{i+1}" for i in range(m)]
    )

elif sens_category == "Demand":
    selected_index = st.selectbox(
        "Select Demand Variable",
        [f"D{i+1}" for i in range(n)]
    )

else:
    selected_index = st.selectbox(
        "Select Capacity Variable",
        [f"E{i+1}" for i in range(k)]
    )
delta = st.number_input(
    "Change Amount (+/-)",
    value=0.0
)
# ---------------------------
# STEP 5: Solve Button
# ---------------------------
if st.button(" Solve Problem"):
    
    # ---------------------------
    # Convert input
    # ---------------------------
    objective_matrices = convert_to_3d(objectives, m, n, k)

    # ---------------------------
    # Apply Sensitivity Perturbation
    # ---------------------------
    new_supply = supply.copy()
    new_demand = demand.copy()
    new_capacity = capacity.copy()

    # Extract numeric index
    idx = int(selected_index[1:]) - 1

    if sens_category == "Supply":
        new_supply = perturb_parameter(supply, idx, delta)

    elif sens_category == "Demand":
        new_demand = perturb_parameter(demand, idx, delta)

    elif sens_category == "Capacity":
        new_capacity = perturb_parameter(capacity, idx, delta)

    # ---------------------------
    # Balance Check
    # ---------------------------
    if abs(sum(new_supply) - sum(new_demand)) > 0.001:
        st.error("❌ Problem became unbalanced after perturbation")

    elif abs(sum(new_supply) - sum(new_capacity)) > 0.001:
        st.error("❌ Capacity mismatch after perturbation")

    else:

        # ---------------------------
        # ORIGINAL SOLUTION
        # ---------------------------
        L, U, payoff = compute_L_U(
            objective_matrices,
            supply,
            demand,
            capacity
        )

        st.subheader(" Payoff Matrix")

        df_payoff = payoff_to_table(payoff)
        st.dataframe(df_payoff)

        # ---------------------------
        # L & U TABLE
        # ---------------------------
        df_bounds = bounds_to_table(L, U)

        st.subheader(" Objective Bounds (L & U)")
        st.dataframe(df_bounds)

        # Metric Cards
        cols = st.columns(len(L))

        for i in range(len(L)):
            cols[i].metric(
                label=f"Obj {i+1}",
                value=f"L: {L[i]}",
                delta=f"U: {U[i]}"
            )

        # ---------------------------
        # ORIGINAL FUZZY SOLUTION
        # ---------------------------
        solution, lambda_val = solve_fuzzy(
            objective_matrices,
            supply,
            demand,
            capacity,
            L,
            U
        )

        # ---------------------------
        # ORIGINAL TABLE OUTPUT
        # ---------------------------
        df_solution = solution_to_table(solution)

        st.subheader(" Original Transportation Plan")
        st.dataframe(df_solution)

        # ---------------------------
        # ORIGINAL OBJECTIVE VALUES
        # ---------------------------
        obj_values = compute_objective_values(
            solution,
            objective_matrices
        )

        st.subheader(" Original Objective Values")

        for i, val in enumerate(obj_values):
            st.write(f"Objective {i+1}: {val}")

        # ---------------------------
        # ORIGINAL VISUALIZATION
        # ---------------------------
        st.subheader(" Objective Comparison")

        fig1 = plot_objectives(obj_values)
        st.plotly_chart(fig1, key="original_objectives")

        st.subheader(" Original Lambda Value")
        st.write("λ =", lambda_val)

        fig2 = plot_lambda(lambda_val)
        st.plotly_chart(fig2, key="original_lambda")

        # =====================================================
        # SENSITIVITY ANALYSIS
        # =====================================================

        st.header(" Sensitivity Analysis Results")

        # ---------------------------
        # Solve Modified Problem
        # ---------------------------
        new_L, new_U, new_payoff = compute_L_U(
            objective_matrices,
            new_supply,
            new_demand,
            new_capacity
        )

        new_solution, new_lambda = solve_fuzzy(
            objective_matrices,
            new_supply,
            new_demand,
            new_capacity,
            new_L,
            new_U
        )

        # ---------------------------
        # Modified Objective Values
        # ---------------------------
        new_obj_values = compute_objective_values(
            new_solution,
            objective_matrices
        )

        # ---------------------------
        # Comparison Table
        # ---------------------------
        comparison_df = pd.DataFrame({
            "Metric": ["Lambda"] + [f"Obj {i+1}" for i in range(len(obj_values))],

            "Original": [lambda_val] + obj_values,

            "Modified": [new_lambda] + new_obj_values
        })

        st.subheader(" Sensitivity Comparison")

        st.dataframe(comparison_df)

        # ---------------------------
        # Modified Transportation Plan
        # ---------------------------
        new_df_solution = solution_to_table(new_solution)

        st.subheader(" Modified Transportation Plan")
        st.dataframe(new_df_solution)

        # ---------------------------
        # Modified Visualization
        # ---------------------------
        st.subheader(" Modified Objective Comparison")

        fig3 = plot_objectives(new_obj_values)
        st.plotly_chart(fig3, key="modified_objectives")

        st.subheader(" Modified Lambda Value")

        fig4 = plot_lambda(new_lambda)
        st.plotly_chart(fig4, key="modified_lambda")