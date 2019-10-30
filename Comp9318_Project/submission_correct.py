from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import helper



def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {}

    ##..................................#

    test_set = [line.strip().split(' ') for line in open(test_data)]

    org_set = [line.strip().split(' ') for line in open(test_data)]

    strategy_instance = helper.strategy()

    def get_freq_of_tokens(sms):
        tokens = {}
        for token in sms:
            # if(len(token)<2):continue
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

    x_train[np.where(x_train > 0)] = 1

    y_train = encoder.fit_transform(y_train)

    # dict_words = {dict_vectorizer.feature_names_.index(values) :values for values in dict_vectorizer.feature_names_}

    dict_x_train = dict(zip(dict_vectorizer.vocabulary_.values(), dict_vectorizer.vocabulary_.keys()))

    parameters = {"gamma": "auto", "C": 20.0, "kernel": "linear", "degree": 1, "coef0": 0.0}
    #parameters = {"gamma": 8 / 100000, "C": 1000, "kernel": "rbf", "degree": 3, "coef0": 0.0}

    clf = strategy_instance.train_svm(parameters, x_train, y_train)

    sigma = clf.coef_.tolist()[0]
    '''
    index_list = [ e for e in dict_x_train.keys()]
    sigma_index_list = [ (index_list[e], sigma[e]) for e in range(len(sigma))]

    print(sigma)
    print(index_list)
    print(sigma_index_list)
    sigma_index_list.sort(key=lambda x: x[1], reverse=True)
    '''
    feature_list = [(dict_vectorizer.feature_names_[e], sigma[e]) for e in range(len(sigma))]

    feature_list.sort(key=lambda x: x[1], reverse=True)


    #print(sigma_index_list)

    #print(clf.score(_test, [1] * 200))

    #traverse_list = [ e[0] for e in sigma_index_list]

    for e in range(len(test_set)):
        record = set(org_set[e])
        for el in feature_list:
            sample = set(test_set[e])
            if len((set(record) - set(sample)) | (set(sample) - set(record))) >= 20: break
            if el[0] in test_set[e]:
                test_set[e] = list(filter(lambda x:x!=el[0], test_set[e]))


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
