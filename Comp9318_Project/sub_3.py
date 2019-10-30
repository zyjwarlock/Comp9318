from sklearn.feature_extraction.text import CountVectorizer

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


    vectorizer = CountVectorizer()
    corpusTotoken = vectorizer.fit_transform(corpus)

    _y_train = [0] * len(strategy_instance.class0) + [1] * len(strategy_instance.class1)

    _x_train = corpusTotoken.todense()

    _test = vectorizer.transform([" ".join(e) for e in test_set]).todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 10.0, "kernel": "linear", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

    sigma = clf.coef_.tolist()[0]

    sigma_list_p = [e for e in sigma if e > 0]
    sigma_list_p.sort(reverse=False)

    sigma_list_n =[e for e in sigma if e < 0]
    sigma_list_n.sort(reverse=False)

    flag = True
    while flag:
        flag = False
        if (len(sigma_list_p) > 0):
            cur_p = sigma_list_p.pop()
            index_p = sigma.index(cur_p)
            max_voc = dict_x_train[index_p]
            for e in range(len(test_set)):
                record = set(org_set[e])
                sample = set(test_set[e])
                if len((set(record) - set(sample)) | (set(sample) - set(record))) < 20:
                    flag=True
                    if (max_voc not in test_set[e]): continue
                    for el in sigma_list_n:
                        if dict_x_train[sigma.index(el)] in test_set[e]:
                            test_set[e].remove(dict_x_train[sigma.index(el)])
                            _sample = set(test_set[e])
                            if(len((set(record) - set(_sample)) | (set(_sample) - set(record))) < 20):
                                test_set[e].append(max_voc)



    f_new = open('modified_data.txt', 'w', encoding='utf-8')

    for line in test_set:
        str = " ".join(line) + "\n"
        f_new.write(str)

    f_new.close()

    ##..................................#

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...

    ## You can check that the modified text is within the modification limits.
    modified_data = './modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance  ## NOTE: You are required to return the instance of this class.
