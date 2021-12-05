from oswegonlp.constants import OFFSET
from collections import Counter
from torch.autograd import Variable
import numpy as np
from collections import defaultdict, Counter
import torch

# deliverable 6.1
def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the k features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''

    newDict = defaultdict(float)
    for i in weights:
        if i[0] == label:
            newDict[i] += weights.get(i)

    highest_weights = sorted(newDict.items(), key=lambda x:x[1],reverse=True)
    highest_weights = highest_weights[:k]


    return highest_weights


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,q=5):
    '''
    Return the k words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    model([0][0][0]) -> gets the first thing in array
    '''
    vocab = sorted(vocab)

    weights = model.state_dict()['Linear.weight']
    label_weights = {}
    new_weights = {}
    wordlist = []

    for i, val in enumerate(weights[label_set.index(label)]):
        label_weights[i] = val.item()

    for k,v in sorted(label_weights.items()):
        new_weights[vocab[k]] = v

    highest_weights = sorted(new_weights.items(), key=lambda x: x[1], reverse=True)

    for word, val in highest_weights:
        wordlist.append(word)

    wordlist = wordlist[:q]
    wordlist.reverse()

    return wordlist
