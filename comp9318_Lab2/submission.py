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

def sse(arr):

    if len(arr) == 0: # deal with arr == []
        return 0.0

    _avg = np.average(arr)
    _val = sum( [(e-_avg)**2 for e in arr] )
    return _val, arr


def v_opt_dp(x, num_bins):# do not change the heading of the function

	_matrix = [[2**32 for i in range(len(x))] for j in range(num_bins)]
	_listmatrix = [[[] for i in range(len(x))] for j in range(num_bins)]

	for i in range(num_bins):
		bin = i + 1
		_list = []
		for j in range(len(x)):
			if (num_bins - 1 > (i + j) or len(x) - j < bin):
				_matrix[i][j] = -1
				continue
			if (i == 0):
				_matrix[i][j] = sse(x[j:len(x)])[0]
				_listmatrix[i][j].append(x[j:len(x)])
			else:
				# _matrix[i][j] = min(sse(x[j:e])+_matrix[i-1][e] for e in range(j+1, len(x)))

				for e in range(j, len(x)-i):
					_val = sse(x[j:e+1])
					cost = _val[0] + _matrix[i - 1][e+1]
					costlist = []
					costlist.append(_val[1])
					for _e in _listmatrix[i - 1][e+1]:
						costlist.append(_e)
					if (cost < _matrix[i][j]):
						_matrix[i][j] = cost
						_listmatrix[i][j] = costlist
	return _matrix, _listmatrix[num_bins - 1][0]
