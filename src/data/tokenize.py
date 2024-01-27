from nltk.stem.porter import PorterStemmer
import numpy as np

ignored_characters = {'!', '~', '^', '.', '%', '$', '^', '_'}

stemmer = PorterStemmer()


def tokenize(pattern):
  pattern = stemmer.stem(pattern)

  for i, letter in enumerate(pattern):
    if letter in ignored_characters: pattern.replace(letter, '')

  return [stemmer.stem(word.lower()) for word in pattern]


def bag_of_words(tokens, dictionary):
  bag = np.zeros(len(dictionary), dtype=np.float32)

  for i, w in enumerate(dictionary):
    if w in tokens: bag[i] = 1.0

  return bag
