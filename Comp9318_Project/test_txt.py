from sklearn.feature_extraction.text import CountVectorizer


list_ = ["will. return? to kyrgyzstan soon . I mr. k powell meet frida", "will return to kyrgyzstan soon . mr. powell meet frida"]

list_e = [line.strip().split(' ') for line in list_]

list_el = [" ".join(line) for line in list_e]

vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b')
corpusTotoken = vectorizer.fit_transform(list_el).todense()
print(vectorizer.vocabulary_)

a=1