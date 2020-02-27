#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import time

from .ComputeWeights import weights
from .ConditionalProbsAndSuppesConditions import performConditions
from .DataframeCheck import dataframeCheck
from .DataframeCheck import probCheck, temporalOrderCheck
from .ExportSBNC import export
from .LikelihoodFit import fit
from .PageRank import pageRank
from .ReadDataframes import read
from .WeightedRandomWalk import performRandomWalk


# Mother class
# Manages all the flow of the algorithm
def SBNC(pathDFCategorized, pathOriginalDf, pathOrder, decisionColumn,posColumn, negColumn, inonclusiveThreshold, apparentThreshold):
    print(posColumn,negColumn)
    # Start ticking
    elapsed = time.time()
    # Read the dataframes
    df, temporalOrder = read(pathDFCategorized, pathOrder)
    # For Returns
    probs = None
    scoresDicts = None
    pos = None
    neg = None
    neut = None
    explainable = None
    inco = None
    apparent = None
    disconnectedNodes = None
    scores = None
    invalidMarginal = None
    notDistinguish = None

    #Check if the user inputted right the column names
    if posColumn != negColumn:

        # Dataframe exists
        if df is not None:
            temporalOrder, reason = temporalOrderCheck(df, temporalOrder, decisionColumn,pathOriginalDf)
            # Temporal order exists and its correct
            if reason == "":
                # Check for NaN's, Nulls, Columns/Rows > 1
                reason = dataframeCheck(df)
                print("Dataframe is correct, starting probability computation")
                if reason == "":
                    # Prob computation
                    df, temporalOrder, invalidMarginal, notDistinguish, marginalProbs, jointProbs, reason = probCheck(df,
                                                                                                                      temporalOrder)
                    # Prob checkight delete columns, so we have to check
                    if reason == "":
                        adjacencyMatrix = performConditions(df, temporalOrder, jointProbs, marginalProbs)
                        print("Suppes Conditions done")
                        adjMatrixReconstucted = fit(df, adjacencyMatrix)
                        print("Training done")
                        nodes, probHappened, probNotHappened, substract = weights(df, adjMatrixReconstucted, marginalProbs,
                                                                                  jointProbs)
                        print("Weights done")
                        probs, df, disconnectedNodes = export(df, adjMatrixReconstucted, nodes, substract)
                        print("Graph Reconstructed")
                        if probs is not None:
                            print("SBNC Reconstruction finished with exit")
                            print("Starting discrimination scoring")
                            scores, pos, neg, neut, explainable, inco, apparent = performRandomWalk(df, probs, 1000,
                                                                                                    posColumn, negColumn,
                                                                                                    inonclusiveThreshold,
                                                                                                    apparentThreshold)
                            elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
                        else:
                            reason = "After the reconstruction, the dataset has less than 2 columns"
        else:
            reason = "Error reading the file(s)"
    else:
        reason = "Negative Decision column & Positive Decision column must be different!"

    return reason, df, invalidMarginal, notDistinguish, probs, scores, disconnectedNodes, pos, neg, neut, explainable, inco, apparent, elapsed
