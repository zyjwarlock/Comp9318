
import helper
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

test_data = [line for line in open("test_data.txt")]

corpus = class_0+class_1

vectorizer = CountVectorizer()

corpusTotoken = vectorizer.fit_transform(corpus).todense()

_y_train = [0]*len(class_0) + [1]*len(class_1)

test_set =  vectorizer.transform(test_data).todense()

clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    ('classification', RandomForestClassifier())
])

tcl = clf.fit(corpusTotoken, _y_train)

print(tcl.predict(test_set), end='\n')     #输出预测结果

print(tcl.score)

'''
print(tcl.n_support_ )  #输出正类和负类支持向量总个数

print (tcl.support_)    #输出正类和负类支持向量索引

print (tcl.support_vectors_)  #输出正类和负类支持向量'''


