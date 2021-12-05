from collections import Counter

import numpy as np
import pandas as pd

from oswegonlp.re_tokenizer import RegexTokenizer

# deliverable 1.1
def bag_of_words(text, retok):
    '''
    Count the number of word occurences for each document in the corpus

    :param retok: an instance of RegexTokenizer
    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    
    return Counter(retok.tokenize(text))


# deliverable 1.2
def aggregate_word_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counter = Counter()
    # Put some code here!
    
    return sum(bags_of_words,counter)
    
# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''    
    return set(bow1) - set(bow2)

# deliverable 1.4
def prune_vocabulary(counts,x,threshold):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''


    x_pruned = list()
    vocab = set()
    for i in x:
        prune = Counter()
        for key in i.keys():
            if counts[key] >= threshold:
                vocab.add(key)
                prune[key] = i[key]
        x_pruned.append(prune)

    return x_pruned, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array
    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''

    vocab = sorted(vocab)
    vocabLength = len(vocab)
    bagLength = len(bags_of_words)
    cnt = 0
    grid = np.zeros((bagLength, vocabLength))

    for counters in bags_of_words:
        for word in counters:
            vocabIndex = vocab.index(word)
            grid[cnt,vocabIndex] = counters.get(word)
        cnt = cnt + 1

    return grid
    
### Helper Code ###

def read_data(filename,label='RealOrFake',preprocessor=bag_of_words):
    retok = RegexTokenizer("[A-Za-z']+")
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string,retok) for string in df['Headline'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())