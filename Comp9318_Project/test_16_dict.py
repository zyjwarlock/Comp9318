from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import helper
from sklearn import preprocessing
import numpy as np


#mod_data = [line for line in open("modified_data.txt")]

test_data = [line.strip().split(' ') for line in open("test_data.txt")]
mod_data = [line.strip().split(' ') for line in open("modified_data.txt")]

strategy_instance = helper.strategy()

stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']

def get_freq_of_tokens(sms):
    tokens = {}
    for token in sms:
        #if(len(token)<2):continue
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens

def fillin_trainset(text, category):
    for line in text:
        training_data.append((get_freq_of_tokens(line), category))

def fillin_testset(text, category):
    for line in text:
        testing_data.append((get_freq_of_tokens(line), category))

training_data = []


fillin_trainset(strategy_instance.class0, 0)
fillin_trainset(strategy_instance.class1, 1)

encoder = LabelEncoder()
dict_vectorizer = DictVectorizer(dtype=float, sparse=True)

x_train, y_train = list(zip(*training_data))

x_train = dict_vectorizer.fit_transform(x_train)

x_train = x_train.toarray()

print(x_train)
x_train[np.where(x_train>0)]=1

y_train = encoder.fit_transform(y_train)

#dict_words = {dict_vectorizer.feature_names_.index(values) :values for values in dict_vectorizer.feature_names_}

dict_x_train = dict(zip(dict_vectorizer.vocabulary_.values(), dict_vectorizer.vocabulary_.keys()))


print(dict_vectorizer.vocabulary_)


testing_data = []
fillin_testset(test_data, 1)
x_test, y_test = list(zip(*testing_data))
x_test = dict_vectorizer.transform(x_test)
x_test = x_test.toarray()
x_test[np.where(x_test>0)]=1
y_test = encoder.transform(y_test)

parameters = {"gamma":"auto", "C":10.0, "kernel":"linear", "degree":1,"coef0":0.0}
clf = strategy_instance.train_svm(parameters, x_train, y_train)
print(clf.coef_)
print(clf.score(x_test, y_test))

testing_data = []
fillin_testset(mod_data, 0)
x_test, y_test = list(zip(*testing_data))
x_test = dict_vectorizer.transform(x_test)
x_test = x_test.toarray()
x_test[np.where(x_test>0)]=1
y_test = encoder.transform(y_test)
print(clf.score(x_test, y_test))












'''corpus = class_0 + class_1

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
print(clf.score(mod_set, [0]*200))'''



