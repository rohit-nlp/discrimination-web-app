#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import igraph as p
import pandas as pd
import numpy as np


# Function that perfoms a weighted random walk algorithm
def walk(graph, start, end):
    visited = list()
    visited.append(start)
    # Graph diameter: longest shortest path between any node
    for i in range(graph.diameter()):
        # We store neighbour names
        names = list()
        # For storing neighbour probs
        probs = list()

        # Find the node we visited on each iteration
        node = graph.vs.find(name=visited[-1])

        # If that's the end node, stop
        if node == end:
            return visited
        # For every child this node has
        for neighbor in graph.neighbors(node, mode="OUT"):
            # Get his name
            names.append(graph.vs[neighbor]["name"])
            # Get his edge probability
            # graph.es[edge index] for getting the edge, .es()["edgeprob"] to get his prob
            # graph_get_eid(Node A, Node B) to get the edge that goes from A to B.
            # graph.vs[vertex index] to get the neigh vertex, ["name"] to get his name and then convert it to 'Node B'
            probs.append(graph.es[graph.get_eid(node["name"], graph.vs[neighbor]["name"])]["edgeprob"])

        # Make probs sum up to 1 (needed for choice method)
        probs = [i / sum(probs) for i in probs]
        # Node had no childs
        if len(names) == 0:
            return visited
        # Get the next random visited node
        visited.append(np.random.choice(names, p=probs))
    return visited


# Function that creats an iGraph graph given a dataset
def createGraph(probs):
    # In this case a tuple is ('Node source','Node target','edge prob')
    # i.e. ('sex_Female','workclass_Private',0.06)
    return p.Graph.TupleList([tuple(x) for x in probs.values], directed=True, edge_attrs=["edgeprob"])


# Method counts how many times we end up in a negative or positive decision for a given node
def count(graph, nIter, node, posName, negName):
    # Positive and Negative Scores
    posCount = []
    negCount = []
    # Number of walks that didnt end on a negative or positive decision
    inconclusiveWalks = 0
    # For storing intermediate nodes
    intermediate = list()
    # This number usually is 1000. Doesnt change much the result (as seen in my paper)
    for i in range(nIter):
        # Start walking, with end being pos or neg name
        countPos = walk(graph, node, posName)
        countNeg = walk(graph, node, negName)
        # Count positive walks
        if countPos[-1] == posName:
            size = len(countPos)
            posCount.append(size - 1)
            # Direct edge, no intermediate node (first position is the own start node)
            if size == 2:
                intermediate.append("None")
            else:
                intermediate.append(countPos[-2])
        # Count negative walks
        elif countNeg[-1] == negName:
            size = len(countNeg)
            negCount.append(size - 1)
            # Direct edge, no intermediate node (first position is the own start node)
            if size == 2:
                intermediate.append("None")
            else:
                intermediate.append(countNeg[-2])
        # Count walks that lead to a leaf node
        else:
            inconclusiveWalks += 1
    # If intermediate node is empty, at least show a '-'
    if len(intermediate) == 0: intermediate = ["-"]
    return posCount, negCount, inconclusiveWalks, intermediate


# Method that collects the result from count() and creates the Dataset
def performRandomWalk(df, probs, nIter, posName, negName, indThr, diff):
    # Lists
    negScores = list()
    posScores = list()
    avgPos = list()
    avgNeg = list()
    inter = list()
    var = list()
    veredicts = list()
    veredictsPie = list()
    inconclusiveScores = list()

    # For each node except the decision nodes
    columns = [i for i in df.columns if i != posName and i != negName]
    graph = createGraph(probs)
    # Compute the random walk
    for i in columns:
        # a string for the pieChart so he can collect different veredicts. PieChart Veredicts collect any kind of "Apparent", and the table does not.
        veredictPie = ""
        # First thing we collect for the table is the name
        var.append(i)

        # Get times we arrived in a negative and positive decision
        posScore, negScore, inconclusiveScore, intermediate = count(graph, nIter, i, posName, negName)

        # Caution, we cannot divide by zero
        sizePos = len(posScore)
        sizeNeg = len(negScore)

        # If no pos or neg walks, append a zero
        if sizeNeg == 0:
            negScores.append(0)
            avgNeg.append(0)
        if sizePos == 0:
            posScores.append(0)
            avgPos.append(0)

        # Collect negative and positive walks
        if sizePos != 0:
            avgPos.append(round(sum(posScore) / sizePos, 2))
            posScores.append(sizePos / nIter)

        if sizeNeg != 0:
            negScores.append(sizeNeg / nIter)
            avgNeg.append(round(sum(negScore) / sizeNeg, 2))

        # Collect inconclusive scores
        indScore = inconclusiveScore / nIter
        inconclusiveScores.append(indScore)

        # Compute a veredict
        if indScore >= indThr:
            veredict = "Too many inconclusive outputs"
        else:
            if abs(posScores[-1] - negScores[-1]) < 0.06:
                veredict = "Neutral variable"
            else:
                # If we arrived here, we have to choose betweeen apparent veredict or total negative or positive veredict
                if posScores[-1] > negScores[-1]:
                    veredict = "Favoritism"  # Maybe it will overwrited later
                else:
                    veredict = "Negative Discrimination"
                # If the difference is too small, given inconclusive score we choose between apparent negative or apparent positive
                if ((posScores[-1] - indScore < diff) and (negScores[-1] < indScore)) or (
                        (negScores[-1] - indScore < diff) and posScores[-1] < indScore):
                    veredict = "Apparent " + veredict
                    veredictPie = "Apparent discrimination"

        # Choosing the most common intermediate node
        interName = max(set(intermediate), key=intermediate.count)

        # If this node exists
        if interName != "None" and interName != "-":
            # Get him in percentage form
            probInter = (intermediate.count(interName) / len(intermediate)) * 100

            # If not, shows as 100.0% and i dont want the coma
            if probInter != 100:
                strProbInter = "%.2f" % probInter
            else:
                # or int or %.0... etc
                strProbInter = 100
            inter.append(interName + ": " + str(strProbInter) + "% of times")

            # If we have an intermediate node in more than 50% of the walks, we show it to the user
            if probInter >= 50:
                if veredict == "Favoritism":
                    veredict = "Explainable/Conditional favoritism because of " + interName
                    veredictPie = "Explainable/Conditional Discrimination"
                elif veredict == "Negative Discrimination":
                    veredict = "Explainable/Conditional discrimination because of " + interName
                    veredictPie = "Explainable/Conditional Discrimination"

        else:
            inter.append(interName)

        # Veredict for pie
        veredicts.append(veredict)
        if veredictPie == "":
            veredictsPie.append(veredict)
        else:
            veredictsPie.append(veredictPie)

    # Scores as the final dataset with all the columns collected
    scores = pd.DataFrame(
        {'Name': var, 'Positive Score': posScores, 'Avg. Positive Steps': avgPos, 'Negative Score': negScores,
         'Avg. Negative Steps': avgNeg, 'Intermediate Node': inter, 'Inconclusive Score': inconclusiveScores,
         'Veredict': veredicts},
        columns=['Name', 'Positive Score', 'Avg. Positive Steps', 'Negative Score', 'Avg. Negative Steps',
                 'Intermediate Node', 'Inconclusive Score', 'Veredict'])
    pos, neg, neut, explainable, inco, apparent = makePie(pd.DataFrame({'veredict': veredictsPie}))
    return scores, pos, neg, neut, explainable, inco, apparent


# Function that creates 6 integers, each for every "discrimination type" that will be used for the pie chart
def makePie(veredict):
    pos = 0
    neg = 0
    neut = 0
    inco = 0
    apparent = 0
    explainable = 0

    scoresCount = veredict['veredict'].value_counts()
    # We don't know the possible order (The resulting object will be in descending order so that the first element is the most frequently-occurring element). But what var?)
    for i in scoresCount.keys():
        if i == "Explainable/Conditional Discrimination":
            explainable = scoresCount[i]
        elif i == "Favoritism":
            pos = scoresCount[i]
        elif i == "Negative Discrimination":
            neg = scoresCount[i]
        elif i == "Neutral variable":
            neut = scoresCount[i]
        elif i == "Too many inconclusive outputs":
            inco = scoresCount[i]
        else:  # Apparent discrimination
            apparent = scoresCount[i]
    return pos, neg, neut, explainable, inco, apparent
