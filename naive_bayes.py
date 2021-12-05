import functools
import itertools

from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation

import numpy as np
from collections import defaultdict, Counter

# deliverable 3.1
def corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict
    """

    corpusList = []
    newcorpusList = []
    
    merged_list = tuple(zip(y, x))

    for i in merged_list:
        if i[0] == label:
            corpusList.append(i)

    for i in corpusList:
        newcorpusList.append(i[1])

    corpus = sum(newcorpusList,Counter())

    return corpus


# deliverable 3.2
def estimate_pxy(x,y,label,alpha,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param alpha: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word
    '''

    dict1 = defaultdict(float)
    totalTokenCount = 0
    list1 = []
    corpus = corpus_counts(x, y, label)

    for i in corpus:
        totalTokenCount = totalTokenCount + corpus.get(i)
        list1.append(i)

    for i in vocab:
        if i not in list1:
            corpus[i] = 0

    for i in corpus:
        math1 = np.log((alpha + corpus.get(i)) / ((len(vocab) * alpha) + totalTokenCount))
        dict1[i] = math1


    return dict1


# deliverable 3.3
def estimate_nb(x,y,alpha):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """

    vocab = []
    fakeCnt = 0
    realCnt = 0



    x1 = sum(x,Counter())

    for i in x1:
        vocab.append(i)

    vocab = set(vocab)

    fake_estimate_pxy = estimate_pxy(x,y,'fake',alpha,vocab)
    real_estimate_pxy = estimate_pxy(x,y,'real', alpha, vocab)

    estimate_nb_dict = defaultdict(float)


    for i in y:
        if i == 'fake':
            fakeCnt = fakeCnt + 1
        else:
            realCnt = realCnt + 1

    f = fakeCnt/(realCnt + fakeCnt)
    r = realCnt/(fakeCnt + realCnt)

    estimate_nb_dict['fake', OFFSET] = np.log(f)
    estimate_nb_dict['real', OFFSET] = np.log(r)

    for i in fake_estimate_pxy:
        estimate_nb_dict['fake', i] = fake_estimate_pxy.get(i)


    for i in real_estimate_pxy:
        estimate_nb_dict['real', i] = real_estimate_pxy.get(i)


    return estimate_nb_dict


# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float, dict
    y_hat from classifier_base.predict all
    '''

    scores = defaultdict(float)

    for i in alphas:
        aDict = estimate_nb(x_tr, y_tr, i)
        y_hat = classifier_base.predict_all(x_dv, aDict, set(y_dv))
        floatNums = evaluation.acc(y_hat, y_dv)
        scores[i] = floatNums

    best_smoother = max(scores, key=scores.get)

    return  best_smoother, scores
