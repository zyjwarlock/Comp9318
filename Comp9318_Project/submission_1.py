from sklearn.feature_extraction.text import CountVectorizer

import helper
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    parameters={}
    

    ##..................................#


    parameters = {"gamma": "auto", "C": 1.0, "kernel": "linear", "degree": 3, "coef0": 0.0}
    class_0 = [line for line in open("class-0.txt")]

    class_1 = [line for line in open("class-1.txt")]

    test_set = [line for line in open(test_data)]
    '''
    class_0 = []
    class_1 = []
    for e in strategy_instance.class0:
        class_0.append(" ".join(e).replace(",","").replace(".",""))

    for e in strategy_instance.class1:
        class_1.append(" ".join(e).replace(",","").replace(".",""))'''

    corpus = class_0+class_1
    '''
    stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any',
                      'at', 'on', 'in', 'by', 'of',
                      'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my',
                      'our', 'his', 'its', 'their',
                      'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so',
                      'be', 'and', 'or']'''

    vectorizer = CountVectorizer(stop_words=stop_word_list)
    corpusTotoken = vectorizer.fit_transform(corpus)

    _y_train = [0] * len(class_0) + [1] * len(class_1)
    _x_train = corpusTotoken.todense()

    dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 1.0, "kernel": "linear", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, _x_train, _y_train)
    sigma = clf.coef_.tolist()[0]

    f_new = open('modified_data.txt', 'w', encoding='utf-8')
    count = 0


    while True:
        _index = sigma.index(max(sigma) if max(sigma) ** 2 > min(sigma) ** 2 else min(sigma))
        flag = False
        for line in test_data:
            if dict_x_train[_index] in line:
                line = line.replace(dict_x_train[_index], "")
                flag = True
            f_new.write(line)

        if (flag): count += 1
        if (count == 20): break

    f_new.close()

    ##..................................#
    
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
