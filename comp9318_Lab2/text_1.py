## import modules here
import pandas as pd
import numpy as np

def v_opt_dp(x, num_bins):# do not change the heading of the function

    _matrix = [[2**32 for i in range(len(x))] for j in range(num_bins)]
    _listmatrix = [[[] for i in range(len(x))] for j in range(num_bins)]

    for i in range(num_bins):
        bin = i+1
        _list = []
        for j in range(len(x)):
            if(num_bins-1>(i+j) or len(x)-j < bin):
                _matrix[i][j] = -1
                continue
            if(i==0):
                _matrix[i][j] = sse(x[j:len(x)])[0]
                _listmatrix[i][j].append(x[j:len(x)])
            else:
               # _matrix[i][j] = min(sse(x[j:e])+_matrix[i-1][e] for e in range(j+1, len(x)))

                for e in range(j+1, len(x)):
                    _val = sse(x[j:e])
                    cost = _val[0]+_matrix[i-1][e]
                    costlist = []
                    costlist.append(_val[1])
                    for _e in _listmatrix[i-1][e]:
                        costlist.append(_e)
                    if(cost < _matrix[i][j]):
                        _matrix[i][j] = cost
                        _listmatrix[i][j] = costlist
    return _matrix, _listmatrix[num_bins-1][0]


def sse(arr):

    if len(arr) == 0: # deal with arr == []
        return 0.0

    _avg = np.average(arr)
    _val = sum( [(e-_avg)**2 for e in arr] )
    return _val, arr

