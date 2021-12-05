from oswegonlp.constants import OFFSET
import numpy as np

//yooooo!
# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# deliverable 2.1
def make_feature_vector(x,y):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)
    kj,m
    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict
    '''

    x = dict(x)
    newDict = dict()

    newDict[y, OFFSET] = 1
    for i in x:
        newDict[y, i] = x.get(i)



    return newDict


# deliverable 2.2
def predict(x,weights,labels):
    '''
    prediction function

    :param x: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict
    '''

    allLabels = dict({'fake': 0,'real': 0})

    for i in labels:
        aDict = make_feature_vector(x,i)
        for i, j in weights.items():
            for p in labels:
                if i[0] == p and aDict.get(i) != None:
                        allLabels[p] += aDict.get(i) * j

    topLabel = max(allLabels, key=allLabels.get)

    return topLabel, allLabels




def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array
    '''

    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat
