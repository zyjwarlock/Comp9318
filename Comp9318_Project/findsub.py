test_set = [line.strip().split(' ') for line in open('test_data.txt')]

new_set = [line.strip().split(' ') for line in open('newmod92.txt')]

our_set =  [line.strip().split(' ') for line in open('modified_data.txt')]

rank_list = []

for e in range(len(test_set)):
    newrecord = set(new_set[e])
    sample = set(test_set[e])
    ourrecord = set(our_set[e])

    print(e)
    print('our add:   ', set(ourrecord)-set(sample))
    print('his add:   ', set(newrecord) - set(sample))
    print('our del:   ', set(sample) - set(ourrecord))
    print('his del:   ', set(sample) - set(newrecord))



