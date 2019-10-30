from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import libsvm
from sklearn.naive_bayes import BernoulliNB
import helper


modified_data = [line for line in open("modified_data.txt")]

test_data = [line for line in open("test_data.txt")]

strategy_instance = helper.strategy()

corpus = [" ".join(e) for e in strategy_instance.class0]+[" ".join(e) for e in strategy_instance.class1]
#corpus = class_0 + class_1
#corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]


stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']


vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus)

_y_train = [0]*len(strategy_instance.class0) + [1]*len(strategy_instance.class1)
#_y_train = [0]*2 + [1]*1

_x_train = corpusTotoken.todense()

dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

modified_set =  vectorizer.transform(modified_data).todense()

test_set =  vectorizer.transform(test_data).todense()

#_class_1 = vectorizer.transform(class_1).todense()

parameters = {"gamma":"auto", "C":1.0, "kernel":"linear","degree":3,"coef0":0.0}
'''
clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

print(clf.predict(test_set))

print(clf.score(test_set, [1]*200))'''

lclf = LinearSVC(penalty='l1', C=10., dual=False).fit(_x_train, _y_train)

print(lclf.predict(test_set))

print(lclf.score(test_set, [1]*200))

print(lclf.cof_)

'''

lclf = LinearRegression().fit(_x_train, _y_train)

print(lclf.predict(test_set).tolist())

print(lclf.score(test_set, [1]*200))'''
'''
tclf = SVC(kernel='linear')

lsvc = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=True))),
  ('svc', tclf)
])

lsvc = lsvc.fit(_x_train, _y_train)

print(lsvc.predict(test_set))

print(lsvc.score(test_set, [1]*200))'''
