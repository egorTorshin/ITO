import numpy as np


def simplex_method(problem_num, C, A, b, eps=1e-6):
    print(f"Optimization Problem №{problem_num}: ")
    print("max z = ", end="")

    for i in range(len(C)):
        coefficient = C[i]
        if i > 0 and coefficient >= 0:
            print(f" + {coefficient}*x{i + 1}", end="")
        elif coefficient < 0:
            print(f" - {-coefficient}*x{i + 1}", end="")
        else:
            print(f"{coefficient}*x{i + 1}", end="")
    print()

    print("\nSubject to the constraints:")
    for i in range(len(A)):
        for j in range(len(A[i])):
            coefficient = A[i][j]
            if j == 0:
                if coefficient >= 0:
                    print(f"{coefficient}*x{j + 1}", end="")
                else:
                    print(f"- {-coefficient}*x{j + 1}", end="")
            else:
                if coefficient >= 0:
                    print(f" + {coefficient}*x{j + 1}", end="")
                else:
                    print(f" - {-coefficient}*x{j + 1}", end="")
        print(f" <= {b[i]}")
    print()

    def to_tableau(costs, constraints, b):
        tableau = []
        for row in constraints:
            tableau.append(row + [b[len(tableau)]])
        z_row = list(costs) + [0]
        tableau.append(z_row)
        return tableau

    tableau = to_tableau(C.tolist(), A.tolist(), b.tolist())

    def can_improve(tableau):
        z_row = tableau[-1][:-1]
        return any(x > 0 for x in z_row)

    def get_pivot_position(tableau):
        z_row = tableau[-1][:-1]
        column = z_row.index(max(z_row))

        restrictions = [float('inf')] * (len(tableau) - 1)
        for i in range(len(tableau) - 1):
            element = tableau[i][column]
            if element > 0:
                restrictions[i] = tableau[i][-1] / element

        if all(r == float('inf') for r in restrictions):
            return None, None  # No valid pivot

        row = restrictions.index(min(restrictions))
        return row, column

    def pivot_step(tableau, pivot_position):
        pivot_row_index, pivot_column_index = pivot_position
        pivot_value = tableau[pivot_row_index][pivot_column_index]

        tableau[pivot_row_index] = [x / pivot_value for x in tableau[pivot_row_index]]

        for eq_index in range(len(tableau)):
            if eq_index != pivot_row_index:
                multiplier = tableau[eq_index][pivot_column_index]
                tableau[eq_index] = [tableau[eq_index][k] - multiplier * tableau[pivot_row_index][k] for k in
                                     range(len(tableau[eq_index]))]

    while can_improve(tableau):
        pivot_position = get_pivot_position(tableau)
        if pivot_position == (None, None):
            return "unbounded", None
        pivot_step(tableau, pivot_position)

    def is_basic(column):
        return column.count(1) == 1 and column.count(0) == len(column) - 1

    def extract_solution(tableau):
        columns = list(zip(*tableau))
        solutions = [0] * (len(columns) - 1)

        for j in range(len(columns) - 1):
            if is_basic(columns[j]):
                one_index = columns[j].index(1)
                solutions[j] = columns[-1][one_index]
        return solutions

    optimal_solution = extract_solution(tableau)
    optimal_value = tableau[-1][-1]

    return "solved", (optimal_solution, optimal_value)


if __name__ == "__main__":
    def get_result(C, A, b, problem_num):
        res = simplex_method(problem_num, C, A, b)

        if res[0] == "solved":
            print("solver_state: solved")
            optimal_solution = [round(i, 3) for i in res[1][0]]
            print(f"x*: {tuple(optimal_solution)}")
            print(f"Maximum value of z: {-round(res[1][1], 3)}")
        else:
            print("solver_state: unbounded")
            print("The method is not applicable!")
        print('-' * 30)


    #Report Test
    C = np.array([100, 140, 120])
    A = np.array([
        [3, 6, 7],
        [2, 1, 8],
        [1, 1, 1],
        [5, 3, 3]
    ])g
    b = np.array([135, 260, 220, 360])

    get_result(C, A, b, 0)


    #Test 1
    C_1 = np.array([3, 2])
    A_1 = np.array([
        [1, -1],
        [-1, 1]
    ])
    b_1 = np.array([3, 2])

    get_result(C_1, A_1, b_1, 1)

    #Test 2
    C_2 = np.array([9, 10, 16])
    A_2 = np.array([
        [18, 15, 12],
        [6, 4, 8],
        [5, 3, 3]
    ])
    b_2 = np.array([360, 192, 180])

    get_result(C_2, A_2, b_2, 2)

    #Test 3
    C_3 = np.array([4, 5, 4])
    A_3 = np.array([
        [2, 3, 6],
        [4, 2, 4],
        [4, 6, 8]
    ])
    b_3 = np.array([240, 200, 160])

    get_result(C_3, A_3, b_3, 3)

    #Test 4
    C_4 = np.array([3, 5, 4, 6])
    A_4 = np.array([
        [2, 3, 1, 4],
        [1, 2, 3, 1],
        [4, 1, 2, 5]
    ])
    b_4 = np.array([10, 8, 15])

    get_result(C_4, A_4, b_4, 4)

    # Test 5
    C_5 = np.array([2, 12, 17])
    A_5 = np.array([
        [4, -1, 0],
        [2, -2, 3],
        [4, -16, 2]
    ])
    b_5 = np.array([2, 12, 20])

    get_result(C_5, A_5, b_5, 5)


