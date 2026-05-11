def convert_to_3d(objectives_df, m, n, k):
    """
    Convert list of DataFrames into 4D list c[p][i][j][k]
    """
    all_objectives = []

    for df in objectives_df:
        # Create empty 3D matrix
        matrix = [[[0 for _ in range(k)] for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for col in df.columns:
                # Parse column name like D1-E2
                d_part, e_part = col.split("-")
                
                j = int(d_part[1:]) - 1   # destination index
                k_ = int(e_part[1:]) - 1  # mode index
                
                matrix[i][j][k_] = df.iloc[i][col]

        all_objectives.append(matrix)

    return all_objectives