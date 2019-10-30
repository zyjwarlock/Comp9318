import test_1 as  test_1
import math

'''
def fprime(x):
    return 1.0 + math.log(x)

def f(x):
    return x * math.log(x) - 16.0

x = test_1.find_root(f, fprime)

print(x)
print(f(x))
'''




def print_tree(root, indent=0):
    print(' '* indent, root)
    if len(root.children) > 0:
        for child in root.children:
            print_tree(child, indent+4)

#toks = ['1', '[', '2', '[', '3', '4', '5', ']', '6', '[', '7', '8', '[', '9', ']', '10','[', '11', '12', ']', ']', '13', ']']
toks =  ['4',
         '[', '6',
         '[','70','[','1',']','[','2',']',']',
         '[','6','[','70',']','[','60',']',']','[','50','[','3',']','[','4',']',']',']',\
          '[', '80','[','70','[','1',']','[','2',']',']','[','60','[','2',']','[','3',']',']','[','50','[','3',']','[','4',']',']',']']

tt = test_1.make_tree(toks)

print_tree(tt)

'''
t = test_1.Tree('*', [test_1.Tree('1'),
               test_1.Tree('2'),
               test_1.Tree('+', [test_1.Tree('3'),
                          test_1.Tree('4')])])
'''


print(test_1.max_depth(tt))





