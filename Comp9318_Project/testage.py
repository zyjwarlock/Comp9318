from sklearn.feature_extraction.text import CountVectorizer

import helper


def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {}

    ##..................................#

    class_0 = [line for line in open("class-0.txt")]

    class_1 = [line for line in open("class-1.txt")]

    test_set = [line.strip().split(' ') for line in open(test_data)]

    org_set = [line.strip().split(' ') for line in open(test_data)]

    strategy_instance = helper.strategy()



    corpus = class_0 + class_1
    # corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]
    '''
    stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any',
                      'at', 'on', 'in', 'by', 'of',
                      'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my',
                      'our', 'his', 'its', 'their',
                      'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so',
                      'be', 'and', 'or']'''

    vectorizer = CountVectorizer()
    corpusTotoken = vectorizer.fit_transform(corpus)

    _y_train = [0] * len(class_0) + [1] * len(class_1)

    _x_train = corpusTotoken.todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 1.0, "kernel": "linear", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

    clf.predict()