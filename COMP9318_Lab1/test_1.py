## import modules here

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function

    if(x<2):return x

    eps = 1E-7

    left = 0
    right = x
    mid = (left+right)/2
    last = -1

    while(abs(mid-last) > eps):
        if(mid*mid > x):
            right = mid
        else:
            left = mid
        last = mid
        mid = (right+left)/2

    return int(mid)

    pass # **replace** this line with your code


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them


def dx(f, x):
    return abs(0-f(x))


def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function

    val = dx(f, x_0)

    while val > EPSILON:
        x_0 = x_0 - f(x_0) / fprime(x_0)
        val = dx(f, x_0)

    return x_0

    pass # **replace** this line with your code


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens): # do not change the heading of the function

    parenttheses = {']':'['}
    _list_tree = []


    for e in tokens:
        tke = e
        try:
            if isinstance(int(e), int):
                tke = Tree(e)
        except ValueError:
            pass
        if tke not in parenttheses:
            _list_tree.append(tke)
        else:
            try:
                tmp=[]
                while True:
                    el = _list_tree.pop()
                    if el != parenttheses[tke]:
                        tmp.append(el)
                    else:
                        break

                while tmp:
                    _list_tree[-1].add_child(tmp.pop())

            except ValueError:
                return

    return _list_tree[0]
    pass # **replace** this line with your code


def max_depth(root): # do not change the heading of the function

    if root == None:
        return 0
    _Max = 0
    for e in root.children:
        Depth = max_depth(e)
        _Max = _Max if _Max > Depth else Depth
    return  _Max + 1

    pass # **replace** this line with your code