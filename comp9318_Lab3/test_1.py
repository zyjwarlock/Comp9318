## import modules here

################# Question 1 #################

def multinomial_nb(training_data, sms):# do not change the heading of the function

    dict_class= {}
    dict_prior = {}
    dict_count = {}

    list_word = []


    for e in training_data:
        if(e[1] not in dict_class.keys()):
            dict_class[e[1]] = {}
            dict_prior[e[1]] = 0
            dict_count[e[1]] = 0
        dict_prior[e[1]] += 1

        for key, val in e[0].items():
            dict_count[e[1]] += val
            list_word.append(key)
            if (key not in dict_class[e[1]].keys()):
                dict_class[e[1]][key] = val
            else:
                dict_class[e[1]][key] += val


    #dict_prior = {e: float(dict_prior[e])/float(len(training_data)) for e in dict_prior}

    #dict_class = {e: {el : float(dict_class[e][el])/float(dict_count[e]) for el in dict_class[e]} for e in dict_count}

    amount = len(set(list_word))

    dict_test = {}

    dict_post = {key: val for key, val in dict_prior.items()}

    for e in sms:
        if(e not in dict_test.keys()):
            dict_test[e] = 0
        dict_test[e] += 1

    for e in dict_post:
        for key, val in dict_test.items():
            if(key not in list_word): continue
            if(key in dict_class[e].keys()):
                dict_post[e] *= (float(dict_class[e][key]+1)/(dict_count[e]+amount)) **val
            else:
                dict_post[e] *= (float(1)/(dict_count[e]+amount)) **val
        dict_post[e] /= len(training_data)

    return dict_post.values()[1]/dict_post.values()[0]
