import submission as t

'''
input_data = t.read_data('./asset/a_')
output = t.buc_rec_optimized(input_data)
output
'''

#x = [1,3,1,3,1,3]
#num_bins = 3

#x = [3, 1, 18, 11, 13, 17]
#num_bins = 4

x = [3, 1, 18, 11, 13, 17, 5, 7, 19, 30, 27, 4, 8, 7, 9, 20, 10, 14, 9]
num_bins = 6

matrix, bins = t.v_opt_dp(x, num_bins)
print("Bins = {}".format(bins))
print("Matrix =")
for row in matrix:
    print(row)