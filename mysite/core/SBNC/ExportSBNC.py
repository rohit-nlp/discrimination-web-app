import pandas as pd
import numpy as np
import igraph as p


def export(df, adjMatrixReconstucted, nodes,substract):

    source = list()
    target = list()
    probHappenedMinusNotHappened = list()

    for i in range(np.shape(adjMatrixReconstucted)[0]):
        for j in range(np.shape(adjMatrixReconstucted)[1]):
            if adjMatrixReconstucted[i,j] == 1:
                source.append(df.columns[i])
                target.append(df.columns[j])
                posInList = findInListOfLists(nodes[j],i)
                probHappenedMinusNotHappened.append(substract[j][posInList])

    disconnectedNodes = [i for i in df.columns if i not in source and i not in target]

    if len(disconnectedNodes) != len(df.columns):
        print("After the reconstruction, ",len(disconnectedNodes)," variables (nodes) appear completely disconnected: ")
        for i in disconnectedNodes:
            print("'",i,"'")
        print("Those will not be used to compute the discrimination score")

    df = df.drop(disconnectedNodes,axis=1)

    resultDF = pd.DataFrame({'Source':source,'Target':target,"P(e|c)-P(e|not c)":probHappenedMinusNotHappened})

    if (resultDF.shape[0] > 1) and (resultDF.shape[1] > 0):
        #resultDF.to_csv("SBNCResults.csv",sep = ";",index = None)
        #plotGraph(resultDF)
        return resultDF,df
    return None,None


def findInListOfLists(list,element):
    for i,j in enumerate(list):
        if j == element:
            return i
    #That will never happen
    return -1

def plotGraph(probs):

    tuples = [tuple(x) for x in probs.round({'edgeprob': 2}).values]
    gPlot = p.Graph.TupleList(tuples, directed=True, edge_attrs=['edgeprob'])
    layout = gPlot.layout("large")
    p.plot(gPlot, "media/DAG Reconstructed.pdf", layout=layout, bbox=(2480, 3508), vertex_size=100, label_size=50,
           margin=[100, 100, 100, 100])