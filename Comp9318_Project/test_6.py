import helper
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

test_data = [line for line in open("test_data.txt")]

corpus = class_0+class_1

corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus).todense()

_y_train = [0]*len(class_0) + [1]*len(class_1)

_y_train = [0]*1 + [1]*2

test_set =  vectorizer.transform(test_data).todense()

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2, k=2).fit_transform(corpusTotoken, _y_train)

print(X_new)





