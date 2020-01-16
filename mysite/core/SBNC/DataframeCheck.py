#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import numpy as np


# Computes marginal and joint probs to determinate if the dataset joins the necessary condtions for the discrimation analisis.
def probCheck(df, temporalOrder):
    jointProbs, marginalProbs = marginalAndJointProbs(df)

    # Maybe there are invalid columns where all its values are 1 or 0. So we collect the correct ones
    validEvents = marginalProbs[(marginalProbs > 0) & (marginalProbs < 1)]
    invalid = list()

    # And if there are invalid event, we delete them from the dataset and the temporal order table
    if len(validEvents) != len(marginalProbs):

        for i, j in enumerate(marginalProbs):
            if j not in validEvents:
                print("Event '", df.columns[i], "' will be discarded because it has an invalid Marginal Probability")
                invalid.append(df.columns[i])
        df, temporalOrder = deleteCols(df, temporalOrder, invalid)

    # UserInfo is a list that his content will be displayed to the user.
    notDistinguish = list()
    notDistinguishUserInfo = list()
    if (df.shape[0] > 1) and (df.shape[1] > 0):
        # Recalculate
        jointProbs, marginalProbs = marginalAndJointProbs(df)

        # Merge group of events not distinguishable
        for i in range(np.shape(marginalProbs)[0]):
            for j in range(np.shape(marginalProbs)[0]):
                # Not self cause
                if (i != j and df.columns[j] not in notDistinguish and df.columns[i] not in notDistinguish):
                    if (jointProbs[i, j] / marginalProbs[i] == 1) and (jointProbs[i, j] / marginalProbs[j] == 1):
                        # The 2 events are not distinguishable
                        notDistinguish.append(df.columns[j])
                        info = str(df.columns[j]) + " has been merged to " + str(df.columns[i]) + " ;"
                        notDistinguishUserInfo.append(info)
                        print("Event ", df.columns[i], " and ", df.columns[j],
                              " will be merged because they re not distinguishable")
    if notDistinguish:
        deleteCols(df, temporalOrder, notDistinguish)
    if (df.shape[0] > 1) and (df.shape[1] > 0):
        return df, temporalOrder, invalid, notDistinguishUserInfo, marginalProbs, jointProbs, ""

    return None, None, None, None, "After deleting events the dataframe has less than 1 column. Aborting."


# Function that deletes a series of cols from the dataset and Temporal Order, rembember that for temporal Order a column is a row!
def deleteCols(df, temporalOrder, invalid):
    df = df.drop(invalid, axis=1)
    rowIndex = list()
    # And from the Temporal Order table
    for i, j in enumerate(temporalOrder['attribute']):
        if j in invalid:
            rowIndex.append(i)
    temporalOrder = temporalOrder.drop(rowIndex, axis=0)

    temporalOrder.reset_index(inplace=True)

    return df, temporalOrder


# Check for values different than 1 or 0, check if there are NaN's or nulls.
# Also, a correct metric will be summing zeros and ones. This sum should be the Dataframe rows x columns
def dataframeCheck(df):
    if df.isna().values.any() == False and df.isnull().values.any() == False:
        if (df.shape[0] > 1) and (df.shape[1] > 0) and df[df == 0].count().sum() + df[df == 1].count().sum() == \
                df.shape[
                    0] * df.shape[1]:
            return ""
        else:
            return "Dataframe has less than 1 row/column or contains cells with values different than '0' or '1'"
    return "Dataframe contains 'NULL' or 'NA'"


# Check for possible errors in the Temporal Table
def temporalOrderCheck(df, temporalOrder, pos, neg):
    reason = ""
    # Should always have 2 columns
    if temporalOrder.shape[1] == 2:
        temporalOrder.columns = ["attribute", "order"]
        # Should have always the same ammount of rows as columns has the Dataframe
        if temporalOrder.shape[0] == df.shape[1]:
            notGood = [i for i in temporalOrder['attribute'] if i not in df.columns]
            if len(notGood) > 0:
                reason = "Temporal Order contains attributes that aren't on the dataset"
            else:
                # Check pos and neg variable temporal order value
                if temporalOrder[temporalOrder["attribute"] == pos]["order"].iloc[0] != max(temporalOrder['order']) or \
                        temporalOrder[temporalOrder["attribute"] == neg]["order"].iloc[0] != max(
                    temporalOrder['order']):
                    reason = "Positive and negative decision columns should always have the maximum order value"
        else:
            reason = "Temporal Order must have as many rows as variables in the dataset"
    else:
        reason = "Temporal Order columns must be 2"
    return temporalOrder, reason


# Computes marginal and joint probs
def marginalAndJointProbs(df):
    matrix = df.values
    pairCount = np.zeros((df.shape[1], df.shape[1])).astype(float)
    for i in range(np.shape(matrix)[1]):
        for j in range(np.shape(matrix)[1]):
            # Two pair of columns to be multiplied
            val1 = matrix[:, i]
            val2 = matrix[:, j]
            pairCount[i, j] = val1.transpose().dot(val2)
    # Return marginal probs and joint
    return pairCount / np.shape(matrix)[0], np.array(np.diag(pairCount) / df.shape[0])
