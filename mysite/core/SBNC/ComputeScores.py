#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

# Function that gets the adjacency matrix and for every node with edge creates a list of his weights,
# so we have multiple lists of lists
def score(df, adjMatrixReconstucted, marginalProbs, jointProbs):
    # List of lists
    # In this case, list will be [node0,node1,node2...] where node0 = ['name','name','name'], (the same name). Done like this for simple coding in a future
    nodes = [[i] for i in [-1 for j in range(df.shape[1])]]
    probHappened = [[i] for i in [-1 for j in range(df.shape[1])]]
    probNotHappened = [[i] for i in [-1 for j in range(df.shape[1])]]
    # Stores the difference between probHappened - probNotHappened
    substract = [[i] for i in [-1 for j in range(df.shape[1])]]

    for i in range(df.shape[1]):
        count = 0  # Checks if node has more than one edge
        for j in range(df.shape[1]):
            if adjMatrixReconstucted[j, i] == 1:
                if count == 0:
                    nodes[i][0] = j
                    probHappened[i][0] = jointProbs[i, j] / marginalProbs[j]
                    probNotHappened[i][0] = (marginalProbs[i] - jointProbs[i, j]) / (1 - marginalProbs[j])
                    count = count + 1
                else:
                    nodes[i].append(j)
                    probHappened[i].append(jointProbs[i, j] / marginalProbs[j])
                    probNotHappened[i].append((marginalProbs[i] - jointProbs[i, j]) / (1 - marginalProbs[j]))
        if len(probHappened[i]) <= 1:
            substract[i] = [probHappened[i][0] - probNotHappened[i][0]]
        else:
            toInsert = list()
            for k in range(len(probHappened[i])):
                toInsert.append(probHappened[i][k] - probNotHappened[i][k])
            substract[i] = toInsert

    return nodes, probHappened, probNotHappened, substract
