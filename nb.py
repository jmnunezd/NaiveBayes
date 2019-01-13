import pandas as pd


def frequencies(book):
    text = open(book, 'r')
    text = text.read().lower()

    words = text.split()
    words = [words[i].replace('.', '').replace(',', '').replace('.', '').replace('"', '').replace("'", '')
                 .replace('?', '').replace(';', '').replace(':', '').replace('-', '').replace('!', '').replace('…', '')
                 .replace('“', '').replace('”', '').replace('—', '') for i in range(len(words))]

    pre_dict = {'word': [], 'freq': []}
    for word in words:
        if word in pre_dict['word']:
            pass
        else:
            pre_dict['word'].append(word)
            pre_dict['freq'].append(words.count(word))

    df = pd.DataFrame(pre_dict)
    df.sort_values('freq', ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


print(frequencies('hp1.txt'))
