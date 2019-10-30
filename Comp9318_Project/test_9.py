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

stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']

#corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

vectorizer=CountVectorizer(stop_words=stop_word_list)
corpusTotoken=vectorizer.fit_transform(corpus)

_y_train = [0]*len(class_0) + [1]*len(class_1)

#_y_train = [0]*1 + [1]*2

test_set =  vectorizer.transform(test_data).todense()

_x_train = corpusTotoken.todense()

print(corpusTotoken)

print(_x_train)

print(vectorizer.vocabulary_)

parameters = {"gamma":"auto", "C":1.0, "kernel":"linear","degree":3,"coef0":0.0}

strategy_instance = helper.strategy()

clf = strategy_instance.train_svm(parameters, _x_train, _y_train)



from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(clf, threshold=0.16, prefit=True)
X_new = model.transform(corpusTotoken)

print(model)
print(X_new)
print(X_new.shape[0])
print(X_new.shape[1])

X_new = model.transform(corpusTotoken.todense())
print(X_new)



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

X_new = SelectKBest(mutual_info_classif, k=2
                    ).fit_transform(corpusTotoken, _y_train)

print(X_new)