#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import pandas as pd
import numpy as np
import igraph as p


# From the adjacency matrix, list of list of nodes and list of list of differences (substract) create a dataframe
# With 3 columns: From, To and Weight. That is, the final graph
def export(df, adjMatrixReconstucted, nodes, substract):
    source = list()
    target = list()
    probHappenedMinusNotHappened = list()

    # For every edge
    for i in range(np.shape(adjMatrixReconstucted)[0]):
        for j in range(np.shape(adjMatrixReconstucted)[1]):
            if adjMatrixReconstucted[i, j] == 1:
                source.append(df.columns[i])
                target.append(df.columns[j])
                posInList = findInListOfLists(nodes[j], i)
                probHappenedMinusNotHappened.append(substract[j][posInList])

    # Recollect disconected nodes
    disconnectedNodes = [i for i in df.columns if i not in source and i not in target]

    # Drop disconnected nodes
    df = df.drop(disconnectedNodes, axis=1)

    # Create the final dataframe
    resultDF = pd.DataFrame({'Source': source, 'Target': target, "edgeprob": probHappenedMinusNotHappened})

    # Security check, plot the graph and return
    if (resultDF.shape[0] > 1) and (resultDF.shape[1] > 0):
        # resultDF.to_csv("SBNCResults.csv",sep = ";",index = None)
        plotGraph(resultDF)
        return resultDF, df, pd.DataFrame({'Name': disconnectedNodes})
    return None, None, None


# Simple function that returns the position of the element in a list, by name.
# It's done this way so code looks cleaner
def findInListOfLists(list, element):
    for i, j in enumerate(list):
        if j == element:
            return i
    # That will never happen
    return -1


# Function that creates the graph in iGraph, plots it and save it in an image.
# Creates 2 graph, one small and one large
def plotGraph(probs):
    # Graph creation
    tuples = [tuple(x) for x in probs.round({'edgeprob': 2}).values]
    gPlot = p.Graph.TupleList(tuples, directed=True, edge_attrs=['edgeprob'])
    # Graph attributes
    gPlot.vs["label"] = gPlot.vs["name"]
    gPlot.es["label"] = gPlot.es['edgeprob']
    # Image creation and saving
    p.plot(gPlot, "media/Big DAG Reconstructed.png", bbox=(2480, 3508), vertex_size=100, color="#5bc0de", label_size=50,
           margin=[100, 100, 100, 100])
    p.plot(gPlot, "media/DAG Reconstructed.png", color="#5bc0de", margin=100, bbox=(1000, 630))
