import helper
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

import numpy as np
import copy

def get_freq_of_tokens(sms):
    tokens = {}
    for token in sms:
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens

def in_it(n):
    return n not in l_mod_top10

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    strategy_instance=helper.strategy()
    features_and_labels = []
    for i in (strategy_instance.class0):
        tokens = get_freq_of_tokens(i)
        features_and_labels.append((tokens, 0))
    for i in (strategy_instance.class1):
        tokens = get_freq_of_tokens(i)
        features_and_labels.append((tokens, 1))
    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype = int, sparse = True)
    x,y = list(zip(*features_and_labels))
    x = vectorizer.fit_transform(x)
    y = encoder.fit_transform(y)


    parameters={'C':10.0, 'coef0':0.0, 'degree':3, 'gamma':'auto', 'kernel':'linear'}


    #x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.8)


    clf = strategy_instance.train_svm(parameters, x,y)

    coef = clf.coef_
    # print(coef)
    data = coef.data.tolist()
    idx = coef.indices.tolist()

    _list = [(vectorizer.get_feature_names()[idx[e]], data[e]) for e in range(len(data))]



    words = []
    for i in range(len(data)):
        words.append((idx[i],data[i]))
    words = sorted(words, key=lambda x: x[1])
    class0_dict = {}
    class1_dict = {}
    for i in range(len(words)):
        if words[i][1] < 0:
            class0_dict[words[i][0]] = abs(words[i][1])
        else:
            class1_dict[words[i][0]] = abs(words[i][1])

    word_0 = np.array(sorted(class0_dict.items(), key=lambda x: x[1],reverse=1)[:100])
    word_1 = np.array(sorted(class1_dict.items(), key=lambda x: x[1],reverse=1)[:100])
    word_0_index = word_0[:,0].tolist()
    word_1_index = word_1[:, 0].tolist()
    # for i in word_0_index:
    #     print(vectorizer.feature_names_[int(i)])
    # print(word_0)

    class0_weight = []
    class1_weight = []
    # for_now0 = []
    # for i in word_0_index:
    #     for_now0.append(vectorizer.feature_names_[int(i)])
    # print(for_now0)
    # for_now1 = []
    # for i in word_1_index:
    #     for_now1.append(vectorizer.feature_names_[int(i)])
    # print(for_now1)

    with open('modified_data.txt',"w") as modified_data:
        with open(test_data, "r") as test:
            for line in test:
                l = line.strip().split(' ')
                global l_mod_top10
                l = set(l)
                l = list(l)
                l_new = copy.deepcopy(l)
                for j in range(len(l)):
                    if l[j] not in vectorizer.feature_names_ :
                        continue
                    else:
                        index = vectorizer.feature_names_.index(l[j])
                        if index in class0_dict.keys():
                            class0_weight.append((index, class0_dict[index]))
                        if index in class1_dict.keys():
                            class1_weight.append((index, class1_dict[index]))
                class0_weight = sorted(class0_weight,key=lambda x:x[1],reverse = True)
                class1_weight = sorted(class1_weight, key=lambda x: x[1], reverse=True)
                # print(class0_weight)
                # print(class1_weight)
                # print(l_new)
                l_mod = []
                for m in range(len(class1_weight)):
                    if class1_weight[m][0] not in word_0_index: #and class1_weight[m][0] in word_1_index:
                        l_mod.append(vectorizer.feature_names_[int(class1_weight[m][0])])
                if len(l_mod) >= 10:
                    l_mod_top10 = l_mod[:10]
                else:
                    l_mod_top10 = copy.deepcopy(l_mod)

                count = 0
                original_len = len(l_new)
                l_new = list(filter(in_it, l_new))
                diff = original_len-len(l_new)
                # print(diff)
                count_1 = diff
                for n in range(len(l_mod)):
                    if count_1 < 20:
                        for x in range(count, len(word_0_index)):
                            aa = vectorizer.feature_names_[int(word_0_index[x])]
                            if aa not in l_new:
                                l_new.append(aa)
                                count += 1
                                break
                            else:
                                count += 1
                        count_1 += 1
                # print(count_1)
                while count_1 < 20:
                    print(count_1)
                    for x in range(count, len(word_0_index)):
                        if vectorizer.feature_names_[int(word_0_index[x])] not in l:
                            l_new.append(vectorizer.feature_names_[int(word_0_index[x])])
                        count += 1
                    count_1 += 1

                p = " ".join(str(i) for i in l_new)
                modified_data.write(p+'\n')
                # break

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...


    ## You can check that the modified text is within the modification limits.

    # result = clf.predict(modified_data).tolist()
    # print(result.count(1) / len(result))
    modified_data = './modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.



test_data='./test_data.txt'
strategy_instance = fool_classifier(test_data)


# print('Success %-age = {}-%'.format(result))