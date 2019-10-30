## import modules here
import pandas as pd
import numpy as np

def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return (df)

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
		print(_tmp)
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
	_codeList = [bin(i)[2:].zfill(n) for i in range(0, 2**n)w]
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
    pass # **replace** this line with your code