from collections import defaultdict, Counter
from oswegonlp.classifier_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''

    newDict = {}


    prediction = predict(x,weights,labels)


    if prediction[0] != y:
        updateValues = make_feature_vector(x,prediction[0])
        updateValues2 = make_feature_vector(x, y)
        
        newDict = updateValues2
        
        for i in updateValues:
            if i in newDict:
                newDict[i] -= updateValues.get(i)
            else:
                newDict[i] = -updateValues.get(i)
                
    return newDict




# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    updated_weights = defaultdict(float)
    weight_history = []
    p = Counter()
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            # Put some code here!
            updated_weights = perceptron_update(x_i, y_i, weights, labels)

            for i,j in updated_weights.items():
                weights[i] += j

        weight_history.append(weights.copy())

    return weights, weight_history
