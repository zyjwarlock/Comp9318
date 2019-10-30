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

    # corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]


    vectorizer = CountVectorizer(binary=True)
    corpusTotoken = vectorizer.fit_transform(corpus)

    _y_train = [0] * len(strategy_instance.class0) + [1] * len(strategy_instance.class1)

    _x_train = corpusTotoken.todense()

    _test = vectorizer.transform([" ".join(e) for e in test_set]).todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 10.0, "kernel": "linear", "degree": 3, "coef0": 0.0}
    #parameters = {"gamma": 8 / 100000, "C": 1000, "kernel": "rbf", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, _x_train, _y_train)

    print(clf.score(_test, [1]*200))

    sigma = clf.coef_.tolist()[0] #训练返回的w

    sigma_list_p = [e for e in sigma if e > 0] #w>0 即判为1的

    sigma_list_n =[e for e in sigma if e < 0] #w<0 即判为0的

    flag = True
    while flag:#dict_x_train[] 为存放 feature  的字典
        if(len(sigma_list_p) > 0):
            cur = max(sigma_list_p)
            _index = sigma.index(cur)
            flag = False
            for e in range(len(test_set)):
                record = set(org_set[e])
                sample = set(test_set[e])
                if len((set(record) - set(sample)) | (set(sample) - set(record))) < 20:
                    if dict_x_train[_index] in test_set[e]:#当存在于testset时删除
                        test_set[e].remove(dict_x_train[_index])
                    flag = True
            sigma_list_p.pop(sigma_list_p.index(cur))#pop掉w最高的词
        else:
            cur = min(sigma_list_n)
            _index = sigma.index(cur)
            flag = False
            for e in range(len(test_set)):
                record = set(org_set[e])
                sample = set(test_set[e])
                if len((set(record) - set(sample)) | (set(sample) - set(record))) < 20:
                    if cur < 0 and dict_x_train[_index] not in test_set[e]:
                        test_set[e].append(dict_x_train[_index])
                    flag = True
            sigma_list_n.pop(sigma_list_n.index(cur))

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
