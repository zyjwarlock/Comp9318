from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import helper
from sklearn import preprocessing


class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

test_data = [line for line in open("test_data.txt")]

strategy_instance = helper.strategy()

corpus = class_0 + class_1

stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']


'''vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus)'''

#corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

Vectorizer = TfidfVectorizer(norm="l2")
corpusTotoken=Vectorizer.fit_transform(corpus).toarray()

_y_train = [0]*len(class_0) + [1]*len(class_1)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))

_x_train = corpusTotoken#.todense()

_x_train = min_max_scaler.fit_transform(corpusTotoken)




#dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

#modified_set =  vectorizer.transform(modified_data).todense()

#test_set =  vectorizer.transform(test_data).todense()

test_set =  Vectorizer.transform(test_data).toarray()

test_set = min_max_scaler.fit_transform(test_set)


for i in range(1, 10):
    for j in range(1, 10, 2):
        parameters = {"gamma":"auto", "C":0.00001*j*(10**i), "kernel":"linear", "degree":3,"coef0":0.0}
        clf = strategy_instance.train_svm(parameters, _x_train, _y_train)
        print(0.00001*j*(10**i), "  ",clf.score(test_set, [1]*200))
'''
parameters = {"gamma":"auto", "C":1.0, "kernel":"linear", "degree":3,"coef0":0.0}
clf = strategy_instance.train_svm(parameters, _x_train, _y_train)
print(clf.score(test_set, [1]*200))'''





