#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

from .DAGScores import DAG
from .SuppesConditions import verifySuppesConditions


def performConditions(df, temporalOrder, jointProbs, marginalProbs):
    # Conditional probs
    primaFacieModel, primaFacieModelNotHappened = DAG(df, jointProbs, marginalProbs)

    # Suppes conditions and final adjacency matrix
    adjacencyMatrix = verifySuppesConditions(df, primaFacieModel, primaFacieModelNotHappened, temporalOrder)

    return adjacencyMatrix
