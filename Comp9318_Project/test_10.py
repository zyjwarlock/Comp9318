from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import helper


class_0 = [line for line in open("class-0.txt")]

class_1 = [line for line in open("class-1.txt")]

test_data = [line for line in open("test_data.txt")]

corpus = class_0+class_1
#corpus = ["UNC played Duke in basketball", "Duke lost the basketball game", "I ate a sandwich"]


stop_word_list = ['an', 'the', 'is', 'are', 'am', 'was', 'were', 'this', 'that', 'these', 'those', 'some', 'any', 'at', 'on', 'in', 'by', 'of',
                  'for', 'to', 'with', 'as', 'they', 'we', 'you', 'he', 'she', 'it', 'me', 'us', 'him', 'her', 'my', 'our', 'his', 'its', 'their',
                  'mine', 'ours', 'yours', 'hers', 'theirs', 'where', 'what', 'when', 'why', 'how', 'but', 'so', 'be', 'and', 'or']


vectorizer=CountVectorizer(stop_words=stop_word_list)
corpusTotoken=vectorizer.fit_transform(corpus)

_y_train = [0]*len(class_0) + [1]*len(class_1)
#_y_train = [0]*2 + [1]*1

_x_train = corpusTotoken.todense()

dict_x_train = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))

#test_set =  vectorizer.transform(test_data).todense()



parameters = {"gamma":"auto", "C":1.0, "kernel":"linear","degree":3,"coef0":0.0}

strategy_instance = helper.strategy()

clf = strategy_instance.train_svm(parameters, _x_train, _y_train)
sigma = clf.coef_.tolist()[0]

f_new = open('modified_data.txt','w',encoding='utf-8')
count = 0

while True:
    _index = sigma.index(max(sigma) if max(sigma)**2 > min(sigma)**2 else min(sigma))
    flag = False
    for line in test_data:
        if dict_x_train[_index] in line:
            line = line.replace(dict_x_train[_index],"")
            flag = True
        f_new.write(line)

    if(flag): count+=1
    if (count==20): break

f_new.close()
