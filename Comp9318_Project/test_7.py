import helper
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np


corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]

vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus).todense()

_y_train = [0]*1 + [1]*2

test_set =  vectorizer.transform(test_data).todense()

print(corpusTotoken)

print(vectorizer.vocabulary_)

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import numpy as np

sl = SelectKBest(chi2, k=5)

X_new = sl.fit_transform(corpusTotoken, _y_train)

print(X_new)
print(sl.scores_)
print(sl.pvalues_)

