
def sse(arr):

    if len(arr) == 0: # deal with arr == []
        return 0.0

    _avg = np.average(arr)
    _val = sum( [(e-_avg)**2 for e in arr] )
    return _val


def v_opt_dp(x, num_bins):# do not change the heading of the function

	_matrix = [[2**32 for i in range(len(x))] for j in range(num_bins)]

	for i in range(num_bins):
		bin = i + 1
		_list = []
		for j in range(len(x)):
			if (num_bins-1>(i+j) or len(x) - j < bin):
				_matrix[i][j] = -1
				continue
			if (i == 0):
				_matrix[i][j] = sse(x[j:len(x)])
				#_listmatrix[i][j].append(x[j:len(x)])
			else:
				_matrix[i][j] = min(sse(x[j:e])+_matrix[i-1][e] for e in range(j+1, len(x)))


	return _matrix,[]