import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt



'''
def f(x):
    return x**5+2*x**4 +3

xvals = np.arange(-2, 10, 0.01)
yvals = np.array([f(x) for x in xvals])

plt.plot(xvals, yvals)
plt.plot(xvals, 0 * xvals)
plt.show()
'''

'''
def basketsort(arr):
    book = np.zeros(10001)
    for i in arr:
        a = round(i, 7)*10000
        book[int(a)]+=1
    return book

xvals = np.random.random(10001)
yvals = np.array(basketsort(xvals))
'''

xvals = np.random.random(100001)

bucket = np.zeros(100001)
yvals = np.zeros(100001)

for i in xvals:
    a = round(i, 7)*100000
    bucket[int(a)]+=1

for i in range(0, 100001):
    bucket[i] = bucket[i]+bucket[i-1]

for e in xvals[::-1]:
    try:
        yvals[int(bucket[int(e*100000)])] = e
        bucket[int(e*100000)] -= 1
    except IndexError:
        pass




plt.bar(xvals, yvals, 0.005, color='blue')
'''
plt.plot(xvals, yvals)
plt.plot(xvals, 0 * xvals)'''
plt.show()
