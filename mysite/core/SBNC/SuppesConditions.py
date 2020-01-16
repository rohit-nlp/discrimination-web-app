#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import pandas as pd
import numpy as np


# Function that verifies Probability Raising and Temporal Priority Suppes conditions in the dataset
def verifySuppesConditions(df, primaFacieModel, primaFacieModelNotHappened, temporalOrder):
    # Adjacency matrix of the dataset
    adjacencyMatrix = np.full(((np.shape(primaFacieModel)[1]), np.shape(primaFacieModel)[1]), 1).astype(float)
    np.fill_diagonal(adjacencyMatrix, 0)

    # With colnames so we can extract the temporal ordrer from the temporal order table
    adjMatrixWithColNames = pd.DataFrame(adjacencyMatrix, None, df.columns)

    for i in range(np.shape(adjacencyMatrix)[0]):
        for j in range(i, np.shape(adjacencyMatrix)[1]):
            if i != j:

                # Verify temporal priority
                atributeNameI = adjMatrixWithColNames.columns[i]
                atributeNameJ = adjMatrixWithColNames.columns[j]

                # Gets the temporal order of the variable by name
                levelI = temporalOrder[temporalOrder['attribute'] == atributeNameI].iloc[0]['order']
                levelJ = temporalOrder[temporalOrder['attribute'] == atributeNameJ].iloc[0]['order']

                # If both variables have the same Temporal Order, nothing happens. If not, we put a zero 'deleting' the edge
                if levelI > 0 and levelJ > 0:
                    if levelI > levelJ: adjacencyMatrix[i, j] = 0
                    if levelJ > levelI: adjacencyMatrix[j, i] = 0
                else:
                    # Will never happen
                    print("Err in TemporalOrder Dataframe")

    # Verify Probability Raising
    for i in range(np.shape(adjacencyMatrix)[0]):
        for j in range(i, np.shape(adjacencyMatrix)[1]):
            if i != j:
                # P(j|i)>P(j|not i) the edge i --> j is valid for temporal priority
                if primaFacieModel[i, j] <= primaFacieModelNotHappened[i, j]: adjacencyMatrix[i, j] = 0
                if primaFacieModel[j, i] <= primaFacieModelNotHappened[j, i]: adjacencyMatrix[j, i] = 0

    return adjacencyMatrix
