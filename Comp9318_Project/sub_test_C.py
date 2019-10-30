from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helper


def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {}

    ##..................................#

    corpus = [" ".join(e) for e in strategy_instance.class0] + [" ".join(e) for e in strategy_instance.class1]

    test_set = [line.strip().split(' ') for line in open(test_data)]

    org_set = [line.strip().split(' ') for line in open(test_data)]

    strategy_instance = helper.strategy()

    # corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]


    vectorizer = CountVectorizer(token_pattern='\S+')
    corpusTotoken = vectorizer.fit_transform(corpus)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(corpusTotoken)

    _y_train = [0] * len(strategy_instance.class0) + [1] * len(strategy_instance.class1)

    _x_train = tfidf.todense()

    _test = transformer.transform(vectorizer.transform([" ".join(e) for e in test_set])).todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    for e in range(1, 100):
        parameters = {"gamma": "auto", "C": 0.1*e, "kernel": "linear", "degree": 3, "coef0": 0.0}
    #parameters = {"gamma": 8 / 100000, "C": 1000, "kernel": "rbf", "degree": 3, "coef0": 0.0}

        clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

        print(0.1*e, " ", clf.score(_test, [1]*200))

    sigma = clf.coef_[0].tolist()

    feature_list = [(dict_x_train[e], sigma[e]) for e in range(len(sigma))]
