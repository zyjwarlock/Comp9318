## import modules here
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

def partition(df):
	_dictCount = dict()
	for _key, _item in df.iterrows():
		if(not (str(_item.values[0])) in _dictCount.keys()):
			_dictCount[str(_item.values[0])] = _item.values[-1]
		else:
			_dictCount[str(_item.values[0])] += _item.values[-1]
	return _dictCount

def buc_Sr(df, dim, _msg, _result, output):

	if(dim >= df.columns.size-1 or len(df)==0):
		_tmp = _msg + str(_result)
		_list =  [str(x) for x in _tmp.split(',')]
		_row = pd.Series(_list, index=list(output.columns))
		output.loc[len(output)]=_row
		return

	df_cloumn = project_data(df, 0).drop_duplicates()
	_count = partition(df)
	for i in df_cloumn:
		if(_count[str(i)] > 0):
			dtmp = slice_data_dim0(df, i)
			_tmp = _msg + str(i) + ","
			buc_Sr(dtmp, 0, _tmp, _count[str(i)], output)
	buc_Sr(remove_first_dim(df), 0, _msg+"ALL"+",", _result, output)

def get_bitmap(n):
	_codeList = [bin(i)[2:].zfill(n) for i in range(0, 2**n)]
	_codeList.reverse()
	return _codeList


def buc_rec_optimized(df):# do not change the heading of the function
	if(len(df)<1): return df
	output = pd.DataFrame(columns=list(df.columns))

	if(len(df)==1):
		_length = df.columns.size
		_list = get_bitmap(_length-1)
		for e in _list:
			_tmplist=[str(s) for s in df.iloc[0]]
			for i in range(0, len(e)):
				if(not int(list(e)[i])):
					_tmplist[i] = "ALL"
			_row = pd.Series(_tmplist, index=list(output.columns))
			output.loc[_list.index(e)] = _row
	else:
		_result = sum(e for e in list(project_data(df, -1)))
		buc_Sr(df, 0, "", _result, output)
	return output


################# Question 2 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function
    # pass # **replace** this line with your code
    n = len(x)
    matrix = [[-1 for _i in range(n)] for y in range(num_bins)]
    # matrix = np.zeros(shape=(num_bins, n))
    # matrix.fill('-1') # create a matrix
    # print(matrix)

    # matrix[0][n-1] = 0
    i = n - 1
    while i >= num_bins-1:
        avg = np.average(x[i:n])
        sse = sum((x[j]-avg)**2 for j in range(i, n))
        matrix[0][i] = sse
        i -= 1
    if num_bins == 1:
        bins = [x]
        return matrix, bins

    for b in range(1, num_bins):
        # matrix[b][n-1-b] = 0
        i = n - b - 1
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
    temp = temp[:(num_bins-1)] #top (num_bins-1) diffs
    bins = []
    temp1 = [] # index list
    for m in range(0, len(temp)):
        idx = maxDiff.index(temp[m])
        if not idx+1 in temp1:
            temp1.append(idx+1)
        else:
            idx = maxDiff[idx+1:].index(temp[m]) + idx+1
            temp1.append(idx+1)
    temp1.sort()
    bins.append(x[:temp1[0]])
    for ib in range(1, len(temp1)):
        bins.append(x[temp1[ib-1] : temp1[ib]])
    bins.append(x[temp1[-1]:])
    return matrix, bins