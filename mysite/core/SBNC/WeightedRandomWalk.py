import igraph as p
import pandas as pd
import numpy as np


def walk(graph, start, end):
    visited = list()
    visited.append(start)
    for i in range(graph.diameter()):
        names = list()
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

def createGraph(probs):
    #In this case a tuple is ('Node source','Node target','edge prob')
    #i.e. ('sex_Female','workclass_Private',0.06)
    return p.Graph.TupleList([tuple(x) for x in probs.values],directed=True,edge_attrs=["edgeprob"])

#Method counts how many times we end up in a negative or positive decision for a given node
def count(graph,nIter,node,posName,negName):
    posCount = []
    negCount = []
    neutralCount = 0
    intermediate = list()
    for i in range(nIter):
        sizeP = 0
        sizeN = 0
        countPos = walk(graph,node,posName)
        countNeg = walk(graph, node, negName)
        if countPos[-1] == posName:
            size = len(countPos)
            posCount.append(size-1)
            if size == 2:
                intermediate.append("None")
            else:
                intermediate.append(countPos[-2])
        elif countNeg[-1] == negName:
            size = len(countNeg)
            negCount.append(size-1)
            if size == 2:
                intermediate.append("None")
            else:
                intermediate.append(countNeg[-2])
        else:
            neutralCount += 1
    if len(intermediate) == 0: intermediate = ["-"]
    return posCount,negCount,neutralCount,intermediate


def performRandomWalk(df, probs, nIter, posName, negName,indThr,diff):
    negScores = list()
    posScores = list()
    avgPos = list()
    avgNeg = list()
    inter =list()
    var = list()
    veredicts = list()
    veredictsPie = list()
    neutralScores = list()
    # For each Node
    columns = [i for i in df.columns if i != posName and i != negName]
    graph = createGraph(probs)
    for i in columns:
        veredictPie = ""
        var.append(i)
        # Get times we arrived in a negative and positive decision
        posScore, negScore, neutralScore, intermediate = count(graph, nIter, i, posName, negName)
        # Caution, we cannot divide by zero
        sizePos = len(posScore)
        sizeNeg = len(negScore)

        if sizeNeg == 0:
            negScores.append(0)
            avgNeg.append(0)
        if sizePos == 0:
            posScores.append(0)
            avgPos.append(0)
        if sizePos != 0:
            avgPos.append(round(sum(posScore) / sizePos,2))
            posScores.append(sizePos / nIter)
            veredict = "Favoritism"
        if sizeNeg != 0:
            negScores.append(sizeNeg / nIter)
            avgNeg.append(round(sum(negScore) / sizeNeg,2))
            veredict = "Negative Discrimination"
        indScore = neutralScore / nIter
        neutralScores.append(indScore)
        if indScore >= indThr:
            veredict = "Too many inconclusive outputs"
        else:
            if abs(posScores[-1] - negScores[-1]) < 0.06:
                veredict= "Neutral variable"
            else:
                if ((posScores[-1] - indScore < diff) and (negScores[-1] < indScore)) or ((negScores[-1] - indScore < diff) and posScores[-1] < indScore):
                    veredict = "Apparent "+veredict
                    veredictPie = "Apparent discrimination"

        interName = max(set(intermediate), key=intermediate.count)

        if interName != "None" and interName != "-":
            probInter = (intermediate.count(interName)/len(intermediate)) * 100
            #If not, shows as 100.0% and i dont want the coma
            if probInter != 100:
                probInter = "%.2f" % probInter
            else:
                #or int or %.0... etc
                probInter = 100
            inter.append(interName+": "+str(probInter)+"% of times")
            if veredict == "Favoritism":
                veredict = "Explainable/Conditional favoritism because of "+ interName
                veredictPie = "Explainable/Conditional Discrimination"
            elif veredict == "Negative Discrimination":
                veredict = "Explainable/Conditional discrimination because of "+ interName
                veredictPie = "Explainable/Conditional Discrimination"
        else: inter.append(interName)
        veredicts.append(veredict)
        if veredictPie == "":
            veredictsPie.append(veredict)
        else:
            veredictsPie.append(veredictPie)

    # Scores as dict then this dict sort it by value
    scores = pd.DataFrame(
        {'Name': var, 'Positive Score': posScores, 'Avg. Positive Steps': avgPos, 'Negative Score': negScores,
         'Avg. Negative Steps': avgNeg, 'Intermediate Node':inter, 'Inconclusive Score': neutralScores, 'Veredict':veredicts},
        columns=['Name', 'Positive Score', 'Avg. Positive Steps', 'Negative Score', 'Avg. Negative Steps','Intermediate Node','Inconclusive Score','Veredict'])
    pos,neg,neut,explainable,inco,apparent = makePie(pd.DataFrame({'veredict':veredictsPie}))
    return scores,pos,neg,neut,explainable,inco,apparent

def makePie(veredict):

    pos = 0
    neg = 0
    neut = 0
    inco = 0
    apparent = 0
    explainable = 0

    scoresCount = veredict['veredict'].value_counts()
    #We don't know the possible order (The resulting object will be in descending order so that the first element is the most frequently-occurring element). But what var?)
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
        else: #Apparent discrimination
            apparent = scoresCount[i]
    return pos,neg,neut,explainable,inco,apparent





