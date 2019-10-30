from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import helper
from sklearn import preprocessing
import numpy as np


class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

mod_data = [line for line in open("modified_data.txt")]

test_data = [line for line in open("test_data.txt")]

strategy_instance = helper.strategy()

stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']


corpus = class_0 + class_1

vectorizer=CountVectorizer(stop_words=stop_word_list)

corpusTotoken=vectorizer.fit_transform(corpus)

_y_train = [0]*len(class_0) + [1]*len(class_1)

_x_train = corpusTotoken.todense()




dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))


test_set =  vectorizer.transform(test_data).todense()

mod_set = vectorizer.transform(mod_data).todense()

_x_train[np.where(_x_train>0)]=1

test_set[np.where(test_set>0)] = 1

mod_set[np.where(mod_set>0)] = 1

parameters = {"gamma":"auto", "C":20.0, "kernel":"linear", "degree":1,"coef0":500.0}
clf = strategy_instance.train_svm(parameters, _x_train, _y_train)
print(clf.score(test_set, [1]*200))
print(clf.score(mod_set, [0]*200))


