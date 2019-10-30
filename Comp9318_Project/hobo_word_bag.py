import operator

def get_word_bag(path):
    word_bag = {}
    for line in open(path):
        words = line.split()
        for word in words:
            word = word.strip()
            word = word.strip('\t')
            word = word.strip('\n')
        for word in words:
            if word in word_bag:
                word_bag[word] += 1
            else:
                word_bag[word] = 1
        # print(words)
        # print(line)
    word_bag_sorted = sorted(word_bag.items(), key = operator.itemgetter(1))
    return word_bag

word_bag_0 = get_word_bag('class-0.txt')
word_bag_1 = get_word_bag('class-1.txt')
word_bag_2 = get_word_bag('test_data.txt')

dictionary = set()

for p in word_bag_0:
    # print(p)
    dictionary.add(p)
for p in word_bag_1:
    dictionary.add(p)

delta = {}

for word in dictionary:
    cnt0 = cnt1 = cnt2 = 0
    if word in word_bag_0:
        cnt0 = word_bag_0[word]
    if word in word_bag_1:
        cnt1 = word_bag_1[word]
    if word in word_bag_2:
        cnt2 = word_bag_2[word]
    print(word, cnt0, cnt1, cnt2, sep = '\t')

# print(sorted(delta.items(), key = operator.itemgetter(1)))
