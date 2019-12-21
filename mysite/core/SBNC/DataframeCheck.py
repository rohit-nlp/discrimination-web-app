import pandas as pd
import numpy as np

def probCheck(df,temporalOrder):

    jointProbs, marginalProbs = marginalAndJointProbs(df)

    # Check for probs equal to zero or equal to 1
    validEvents = marginalProbs[(marginalProbs > 0) & (marginalProbs < 1)]
    notValid = [df.columns[i] for i in marginalProbs if i not in validEvents]
    if notValid:
        for i in notValid:
            if i == 0:
                print("Event ", df.columns[i], " will be discarded because it has a Marginal Probability of 0")
            else:
                print("Event ", df.columns[i], " will be discarded because it has a Marginal Probability of 1")
    # Merge group of events not distinguishable
    for i in range(np.shape(marginalProbs)[0]):
        for j in range(np.shape(marginalProbs)[0]):
            # Not self cause
            if (i != j) and (i not in notValid) and (j not in notValid):
                if (jointProbs[i, j] / marginalProbs[i] == 1) and (jointProbs[i, j] / marginalProbs[j] == 1):
                    # The 2 events are not distinguishable
                    notValid.add(df.columns[j])
                    df = df.rename({df.columns[i]: df.columns[i] + "_and_" + df.columns[j]})
                    # df = df.rename({df.columns[j]:df.columns[i]})
                    print("Event ", i, " and ", j, " will be merged because they re not distinguishable")
    if notValid:
        df = df.drop(notValid, axis=1)
        rowIndex = list()
        for i, j in enumerate(temporalOrder['atribute']):
            if j in notValid:
                rowIndex.append(i)
        temporalOrder = temporalOrder.drop(rowIndex, axis=0)
    if (df.shape[0] > 1) and (df.shape[1] > 0):
        return df, temporalOrder, marginalProbs, jointProbs, ""
    return None,None,None,None,"After deleting events the dataframe has less than 1 column. Aborting."


def dataframeCheck(df,temporalOrder):
    if df.isna().values.any() == False and df.isnull().values.any() == False:
        if (df.shape[0] > 1) and (df.shape[1] > 0) and df[df == 0].count().sum() + df[df == 1].count().sum() == df.shape[
            0] * df.shape[1]:
            return ""
        else:
            return "Dataframe has less than 1 row/column or contains cells with values different than '0' or '1'"
    return "Dataframe contains 'NULL' or 'NA'"


def marginalAndJointProbs(df):
    matrix = df.values
    pairCount = np.zeros((df.shape[1], df.shape[1])).astype(float)
    for i in range(np.shape(matrix)[1]):
        for j in range(np.shape(matrix)[1]):
            val1 = matrix[:, i]
            val2 = matrix[:, j]
            pairCount[i, j] = val1.transpose().dot(val2)
    return pairCount/np.shape(matrix)[0], np.array(np.diag(pairCount) / df.shape[0])
