from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helper


def fool_classifier(test_data):  ## Please do not change the function defination...

    strategy_instance = helper.strategy()
    parameters = {}

    corpus = [" ".join(e) for e in strategy_instance.class0] + [" ".join(e) for e in strategy_instance.class1]

    test_set = [line.strip().split(' ') for line in open(test_data)]

    org_set = [line.strip().split(' ') for line in open(test_data)]

    strategy_instance = helper.strategy()

    vectorizer = CountVectorizer(token_pattern='\S+')
    corpusTotoken = vectorizer.fit_transform(corpus)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(corpusTotoken)

    _y_train = [0] * len(strategy_instance.class0) + [1] * len(strategy_instance.class1)

    _x_train = tfidf.todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 1.0, "kernel": "linear", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

    sigma = clf.coef_[0].tolist()

    feature_list = [(dict_x_train[e], sigma[e]) for e in range(len(sigma))]

    feature_list_pos = [e for e in feature_list if e[1] > 0]  # 1
    feature_list_nag = [e for e in feature_list if e[1] < 0]  # 0

    feature_list_pos.sort(key=lambda x: x[1], reverse=True)
    feature_list_nag.sort(key=lambda x: x[1], reverse=False)

    for e in range(len(test_set)):
        record = set(org_set[e])
        for el in feature_list_pos:
            sample = set(test_set[e])
            if len((set(record) - set(sample)) | (set(sample) - set(record))) >= 15: break
            if el[0] in test_set[e]:
                test_set[e] = list(filter(lambda x: x != el[0], test_set[e]))

    for e in range(len(test_set)):
        record = set(org_set[e])
        for el in feature_list_nag:
            sample = set(test_set[e])
            if len((set(record) - set(sample)) | (set(sample) - set(record))) >= 20: break
            if el[0] not in test_set[e]:
                test_set[e].append(el[0])


    f_new = open("modified_data.txt", 'w', encoding='utf-8')

    for line in test_set:
        str = " ".join(line) + "\n"
        f_new.write(str)

    f_new.close()
    modified_data = './modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance  ## NOTE: You are required to return the instance of this class.
