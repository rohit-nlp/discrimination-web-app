import pandas as pd
import numpy as np


def score(df,adjMatrixReconstucted,marginalProbs,jointProbs):

    nodes= [[i] for i in [-1 for j in range(df.shape[1])]]
    probHappened =[[i] for i in [-1 for j in range(df.shape[1])]]
    probNotHappened = [[i] for i in [-1 for j in range(df.shape[1])]]
    substract = [[i] for i in [-1 for j in range(df.shape[1])]]

    for i in range(df.shape[1]):
        count = 0
        for j in range(df.shape[1]):
            if adjMatrixReconstucted[j,i] == 1:
                if count == 0:
                    nodes[i][0] = j
                    probHappened[i][0] = jointProbs[i, j] / marginalProbs[j]
                    probNotHappened[i][0] = (marginalProbs[i]-jointProbs[i,j])/(1-marginalProbs[j])
                    count = count + 1
                else:
                    nodes[i].append(j)
                    probHappened[i].append(jointProbs[i, j] / marginalProbs[j])
                    probNotHappened[i].append((marginalProbs[i]-jointProbs[i,j])/(1-marginalProbs[j]))
        if len(probHappened[i]) <= 1:
            substract[i]=[probHappened[i][0]-probNotHappened[i][0]]
        else:
            toInsert = list()
            for k in range(len(probHappened[i])):
                toInsert.append(probHappened[i][k]-probNotHappened[i][k])
            substract[i] = toInsert


    return nodes, probHappened,probNotHappened,substract
