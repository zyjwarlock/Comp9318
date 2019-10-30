import helper
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

test_data = [line for line in open("test_data.txt")]

corpus = class_0+class_1

#corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus).todense()

_y_train = [0]*len(class_0) + [1]*len(class_1)

#_y_train = [0]*1 + [1]*2

test_set =  vectorizer.transform(test_data).todense()

print(corpusTotoken)

print(vectorizer.vocabulary_)
'''
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

sl = SelectKBest(chi2, k=5)

X_new = sl.fit_transform(corpusTotoken, _y_train)

print(X_new)
print(sl.scores_)
print(sl.pvalues_)'''





strategy_instance = helper.strategy()

parameters = {"gamma":"auto", "C":1.0, "kernel":"linear","degree":3,"coef0":0.0}

tcl = helper.strategy.train_svm(parameters,corpusTotoken, _y_train)

print(tcl.n_support_ )  #输出正类和负类支持向量总个数

print (tcl.support_)    #输出正类和负类支持向量索引

print (tcl.support_vectors_)  #输出正类和负类支持向量'''

