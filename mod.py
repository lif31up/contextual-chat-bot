from nltk.stem.porter import PorterStemmer
import numpy as np

nonsense = {'!','~','^','.','%','$','^','_'}

stemmer = PorterStemmer()
def tokenize(sentence):
    sentence = stemmer.stem(sentence)

    for i, letter in enumerate(sentence):
        if letter in nonsense: del sentence[i]
    # for if

    return [stemmer.stem(word.lower()) for word in sentence]
# tokenize(): get sentence or a word to make a token

def bag_of_words(tked_sentence, allWords):
    bag = np.zeros(len(allWords), dtype=np.float32)

    for i, w in enumerate(allWords):
        if w in tked_sentence: bag[i] = 1.0
    # for if

    return bag
# bag_of_words(): take tokenized sentence to make a vector
