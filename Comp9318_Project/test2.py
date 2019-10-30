import helper
import numpy as np

_x_train = np.array([[i] for i in range(1247)])

_y_train = [0]*100 + [1]*1147


strategy_instance = helper.strategy()

parameters = {"gamma":"auto", "C":1.0, "kernel":"rbf","degree":3,"coef0":0.0}



print(helper.strategy.train_svm(parameters, _x_train, _y_train))