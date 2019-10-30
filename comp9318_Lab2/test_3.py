# import modules here
import pandas as pd
import numpy as np


################# Question 1 #################

# helper functions
def project_data(df, d):
    # Return only the d-th column of INPUT
    return df.iloc[:, d]

def select_data(df, d, val):
    # SELECT * FROM INPUT WHERE input.d = val
    col_name = df.columns[d]
    return df[df[col_name] == val]

def remove_first_dim(df):
    # Remove the first dim of the input
    return df.iloc[:, 1:]

def slice_data_dim0(df, v):
    # syntactic sugar to get R_{ALL} in a less verbose way
    df_temp = select_data(df, 0, v)
    return remove_first_dim(df_temp)


def buc_rec_optimized(df):# do not change the heading of the function
    pass # **replace** this line with your code


################# Question 2 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function
    # pass # **replace** this line with your code
    n = len(x)
    matrix = [[-1 for _i in range(n)] for y in range(num_bins)]
    # matrix = np.zeros(shape=(num_bins, n))
    # matrix.fill('-1') # create a matrix
    # print(matrix)

    matrix[0][n-1] = 0
    i = n - 2
    while i >= num_bins-1:
        avg = np.average(x[i:n])
        sse = sum((x[j]-avg)**2 for j in range(i, n))
        matrix[0][i] = sse
        i -= 1

    for b in range(1, num_bins):
        matrix[b][n-1-b] = 0
        i = n - b - 2
        while i >= num_bins-1-b:
            cost = []
            c = 1
            while i+c < n and matrix[b-1][i+c] > -1:
                avg = np.average(x[i:i+c])
                sse = sum((x[j] - avg) ** 2 for j in range(i, i+c))
                cost.append(matrix[b-1][i+c] + sse)
                c += 1
            matrix[b][i] = min(cost)
            i -= 1

    # Get bins using MaxDiff
    maxDiff = []
    for d in range(1, n):
        maxDiff.append(abs(x[d] - x[d-1]))
    temp = maxDiff[:]
    temp.sort(reverse=-1)
    temp = temp[:(num_bins-1)]
    bins = []
    for m in range(0, len(temp)):
        idx = maxDiff.index(temp[m])
        temp[m] = idx+1
    temp.sort()
    bins.append(x[:temp[0]])
    for ib in range(1, len(temp)):
        bins.append(x[temp[ib-1] : temp[ib]])
    bins.append(x[temp[-1]:])
    return matrix, bins

