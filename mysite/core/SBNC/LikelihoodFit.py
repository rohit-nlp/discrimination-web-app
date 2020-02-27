#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import pandas2ri


# Function that given a graph perfoms Hill Climbing with BIC for structure learning
def fit(df, adjacencyMatrix):
    # HC functiont needs categorical values, not numbers
    # We just recreate the dataframe with "hit" (1) and "miss" (0)
    categoricalMatrix = np.full(df.shape, "miss")
    dfMatrix = df.values

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if dfMatrix[i, j] == 1:
                categoricalMatrix[i, j] = "hit"

    # Same matrix but with colnames
    # Replacing [] for '.' because of the training library.
    categoricalMatrixDF = pd.DataFrame(categoricalMatrix,
                                       columns=[i.replace("[", ".").replace("]", ".") for i in df.columns])

    # Create a blacklist of edges that are not legal by Suppes theory
    nodeFrom = list()
    nodeTo = list()
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(adjacencyMatrix.shape[1]):
            if i != j:
                if adjacencyMatrix[i, j] == 0:
                    nodeFrom.append(df.columns[i].replace("[", ".").replace("]", "."))
                    nodeTo.append(df.columns[j].replace("[", ".").replace("]", "."))

    # If not, bnlearn says dataset contains Na/NaN
    categoricalMatrixDF = categoricalMatrixDF.astype('category')

    blacklist = pd.DataFrame({'From': nodeFrom, 'To': nodeTo})

    ###R interface for the hc function call
    import rpy2.robjects.packages as rpackages

    # import R's utility package
    utils = rpackages.importr('utils')

    # import bnlearn
    try:
        bnlearn = robjects.packages.importr('bnlearn')
    except RRuntimeError:
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages("bnlearn")
        bnlearn = robjects.packages.importr('bnlearn')

    # For converting pandas datasets to R datasets
    pandas2ri.activate()

    # Conversion to R datasets
    # blacklist.to_csv("blacklist",sep=";",index=False)
    # categoricalMatrixDF.to_csv("matrix",sep=";",index=False)
    blacklist_to_R = pandas2ri.py2ri(blacklist)
    df_to_R = pandas2ri.py2ri(categoricalMatrixDF)

    try:
        # Perform the training
        training = bnlearn.hc(df_to_R, score="bic", blacklist=blacklist_to_R)
    except Exception as e:
        return None, "Invalid symbol on header column name!"
        raise


    # arcs is a matrix with 2 columns (from, to) with the computed arcs
    arcs = training[2]

    # Dic to map column names with index. Remember we did the blacklist with var names instead of numbers.
    # Recovering again the []
    indexColumns = {df.columns[i].replace("[", ".").replace("]", "."): i for i in range(0, len(df.columns))}

    # Compute the adjacency matrix of the trained model
    reconstructed = np.zeros(adjacencyMatrix.shape)
    for i in range(arcs.nrow):
        # arcs.rx(x,y) way to adress an R matrix. R matrix start with index 1 (lame)
        # asarray() used to convert a matrix position into a numpy array
        # asarray()[0] because the node name is located on position 0 of the array
        # indexColumns[] map the node name to node position
        # final result is just reconstructed[52,62]
        # we put a '1' becuase i->j means causation
        reconstructed[
            indexColumns[(np.asarray(arcs.rx(i + 1, 1))[0])], indexColumns[(np.asarray(arcs.rx(i + 1, 2))[0])]] = 1
    return reconstructed
