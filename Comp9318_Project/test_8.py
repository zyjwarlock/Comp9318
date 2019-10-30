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

corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus)

_y_train = [0]*len(class_0) + [1]*len(class_1)

_y_train = [0]*1 + [1]*2

test_set =  vectorizer.transform(test_data).todense()

print(corpusTotoken)

print(vectorizer.vocabulary_)

parameters = {"gamma":"auto", "C":1.0, "kernel":"linear","degree":3,"coef0":0.0}

from sklearn import svm

gamma = parameters['gamma']
C = parameters['C']
kernel = parameters['kernel']
degree = parameters['degree']
coef0 = parameters['coef0']

## Train the classifier...
clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)



from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(clf)
model.fit(corpusTotoken, _y_train)
X_new = model.transform(corpusTotoken.todense())



print(model)
print(X_new)