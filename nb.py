import numpy as np
import operator


def suit(book):
    text = open(book, 'r')
    text = text.read().lower()

    words = text.split()
    words = [words[i].replace('.', '').replace(',', '').replace('.', '').replace('"', '').replace("'", '')
                 .replace('?', '').replace(';', '').replace(':', '').replace('-', '').replace('!', '').replace('…', '')
                 .replace('“', '').replace('”', '').replace('—', '').replace('(', '').replace(')', '')
                 .replace('/', '').replace('—', '') for i in range(len(words))]

    return words


def frequencies(book):
    words = suit(book)

    book_dict = {}
    for word in words:
        if word in book_dict.keys():
            pass
        else:
            book_dict[word] = words.count(word)

    return book_dict


def train(*books):
    train_dict = {}
    for book in books:
        bock_dict = frequencies(book)
        total_words = sum(bock_dict.values())
        train_dict[book] = [bock_dict, total_words]

    return train_dict


def test(trained_dict, book_to_predict, n_sample):
    f = frequencies(book_to_predict)
    sample = np.random.choice(list(f.keys()), n_sample)
    posteriori = {}

    for key in trained_dict.keys():
        book_dict = trained_dict[key][0]
        total_words = trained_dict[key][1]
        prob = []
        for word in sample:
            if word in book_dict:
                fi = book_dict[word] / total_words
                prob.append(fi)
            else:  # This seems to be a very bad approximation to this problem, since cn1 has 4 times more words
                prob.append(1 / total_words)  # than hp1, an alternative must the found

        posteriori[key] = np.prod(prob)

    add = sum(posteriori.values())

    for key in posteriori.keys():
        posteriori[key] = posteriori[key] / add

    max_prob_element = max(posteriori.items(), key=operator.itemgetter(1))[0]

    print('the author of ', book_to_predict, 'is more likely to be the author of', max_prob_element,
          'with a probability of ', posteriori[max_prob_element])

    return posteriori


if __name__ == '__main__':
    model = train(*['hp1.txt', 'cn1.txt', 'lr1.txt'])
    test(model, 'hp2.txt', 10)
    test(model, 'cn2.txt', 10)
    test(model, 'lr2.txt', 10)


# if you increase n_sample you'll get even worst results... needs to found a better approach.
# hp1 has 78670 "unique" words
# cn1 has 322710 "unique" words
# lr1 has 195809 "unique" words
