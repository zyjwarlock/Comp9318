from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["我 来到 北京 清华大学",
          "他 来到 了 网易 杭研 大厦",
          "小明 硕士 毕业 与 中国 科学院",
          "我 爱 北京 天安门"]

# token_pattern指定统计词频的模式, 不指定, 默认如英文, 不统计单字
vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b')
# norm=None对词频结果不归一化
# use_idf=False, 因为使用的是计算tfidf的函数, 所以要忽略idf的计算
transformer = TfidfTransformer(norm=None, use_idf=False)
print(vectorizer.fit_transform(corpus))

tf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print(tf)
word = vectorizer.get_feature_names()
weight = tf.toarray()

for i in range(len(weight)):
    for j in range(len(word)):
        print(word[j], ':', weight[i][j], end=' ', sep='')

    print()