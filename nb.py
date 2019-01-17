import numpy as np
import operator


def suit(book):
    """
    suit cleans the 'words' that are found in a book, this helps to count every single word just once. e.g.
    'kiss,' and 'kiss' content actually the same word, so we remove the ',' and makes everything simple.
    :param book: it's the book we want to clean.
    :return: a list that contains every valid word that appears on the book. Notice that in here words may be repeated.
    """
    text = open(f'books/{book}', 'r')
    text = text.read().lower()

    words = text.split()
    words = [words[i].replace('.', '').replace(',', '').replace('.', '').replace('"', '').replace("'", '')
                     .replace('?', '').replace(';', '').replace(':', '').replace('-', '').replace('!', '')
                     .replace('…', '').replace('“', '').replace('”', '').replace('—', '').replace('(', '')
                     .replace(')', '').replace('/', '').replace('–', '').replace('’', '').replace('‘', '')
                     .replace('*', '').replace('%', '').replace('#', '').replace('=', '').replace('+', '')
                     .replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '')
                     .replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '')
             for i in range(len(words))]

    return words


def frequencies(book):
    """
    For a given book, this function clean the words in the book using the suit function and returns a dictionary,
    that contains every word that appear on the book as a key and it's frequencies of occurrence as a value.
    :param book: it's the book we want to know it's words frequencies.
    :return: a dict that contains every word and it's count.
    """
    words = suit(book)

    book_dict = {}
    for word in words:
        if word in book_dict.keys():
            pass
        else:
            book_dict[word] = words.count(word)

    return book_dict


def train(*books):
    """
    this function saves the dict of frequencies of every book input, this will be useful when we wan't to test a future
    book that is not in the list. This creates a sort of a model.
    :param books: a list of book's names that we want our model to be feed of.
    :return: a dictionary that may become handy when using a test function.
    """
    train_dict = {}
    for book in books:
        bock_dict = frequencies(book)
        total_words = sum(bock_dict.values())
        train_dict[book] = [bock_dict, total_words]

    return train_dict


def test(trained_dict, book_to_predict, n_sample):
    """
    given a model, a book we want to predict it's author and a sample size number we obtain the most probable author
    comparing the frequencies of the words in the model and in a sample of words of the current book.
    :param trained_dict: a model, it's the output of the train function.
    :param book_to_predict: the name of the book we want to predict the author.
    :param n_sample: an integer.
    :return: a print that shows what is the most likely author of the book_to_predict.
    """
    f = frequencies(book_to_predict)
    sample = np.random.choice(list(f.keys()), n_sample)

    print('the sample of words taken from', book_to_predict, 'is: ')
    print(sample)
    print()
    max_words_in_book = 1
    for key in trained_dict.keys():
        max_words_in_book = max(trained_dict[key][1], max_words_in_book)

    posteriori = {}
    for key in trained_dict.keys():
        book_dict = trained_dict[key][0]
        total_words = trained_dict[key][1]
        prob = []
        for word in sample:
            if word in book_dict:
                fi = book_dict[word] / total_words
                prob.append(fi)
            else:
                prob.append(1 / max_words_in_book)
        posteriori[key] = np.prod(prob)

    add = sum(posteriori.values())

    for key in posteriori.keys():
        posteriori[key] = posteriori[key] / (add + 0.00000001)

    max_prob_element = max(posteriori.items(), key=operator.itemgetter(1))[0]

    print('the author of ', book_to_predict, 'is more likely to be the author of', max_prob_element)
    print()
    print(posteriori)
    print()


if __name__ == '__main__':
    model = train(*['hp1.txt', 'cn1.txt', 'lr1.txt'])
    test(model, 'hp2.txt', 50)
    test(model, 'cn2.txt', 50)
    test(model, 'lr2.txt', 50)
    test(model, 'cb.txt', 50)
