import igraph as p
import pandas as pd
import numpy as np


def walk(graph, start, end):
    visited = list()
    visited.append((start, 0))
    for i in range(graph.diameter()):
        names = list()
        probs = list()

        # Find the node we visited on each iteration
        node = graph.vs.find(name=visited[-1][0])

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
        probs = [probs[i] / sum(probs) for i in range(len(probs))]
        # Node had no childs
        if len(names) == 0:
            return visited
        # Get the next random visited node
        visited.append((np.random.choice(names, p=probs), i+1))
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
    for i in range(nIter):
        countPos = walk(graph,node,posName)[-1]
        countNeg = walk(graph, node, negName)[-1]
        if countPos[0] == posName:
            posCount.append(countPos[1])
        elif countNeg[0] == negName:
            negCount.append(countNeg[1])
        else:
            neutralCount += 1
    return posCount,negCount,neutralCount


def performRandomWalk(df, probs, nIter, posName, negName):
    negScores = list()
    posScores = list()
    avgPos = list()
    avgNeg = list()
    var = list()
    neutralScores = list()
    # For each Node
    columns = [i for i in df.columns if i != posName and i != negName]
    graph = createGraph(probs)
    for i in columns:
        # Get times we arrived in a negative and positive decision
        posScore, negScore, neutralScore = count(graph, nIter, i, posName, negName)
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
            avgPos.append(sum(posScore) / sizePos)
            posScores.append(sizePos / nIter)
        if sizeNeg != 0:
            negScores.append(sizeNeg / nIter)
            avgNeg.append(sum(negScore) / sizeNeg)

        neutralScores.append(neutralScore / nIter)
        var.append(i)

    # Scores as dict then this dict sort it by value
    scores = pd.DataFrame(
        {'Name': var, 'Positive Score': posScores, 'Avg. Positive': avgPos, 'Negative Score': negScores,
         'Avg. Negative': avgNeg, 'Neutral Score': neutralScores},
        columns=['Name', 'Positive Score', 'Avg. Positive', 'Negative Score', 'Avg. Negative', 'Neutral Score'])
    pos,neg,neut = makePie(scores)
    return scores,pos,neg,neut

def makePie(scores):

    scores = scores.drop(["Name","Avg. Positive","Avg. Negative"],axis=1)
    scores['max_value'] = scores.idxmax(axis=1)
    scoresCount = scores['max_value'].value_counts()

    pos = 0
    neg = 0
    neut = 0
    #We don't know the possible order (The resulting object will be in descending order so that the first element is the most frequently-occurring element. But what var?)
    for i in scoresCount.keys():
        if i == "Negative Score":
            neg = scoresCount[i]
        elif i == "Positive Score":
            pos = scoresCount[i]
        else:
            neut = scoresCount[i]

    # colors = ["#ffb39c", "#E3D4AD", "#6183A6"]
    # labels = ["Positive", "Negative", "Neutral"]
    # # Create a pie chart
    # fig1, ax1 = plt.subplots(figsize=(10, 10))
    # patches, texts, autotexts = ax1.pie(
    #
    #     scoresCount,
    #
    #     labels=list(labels),
    #     # with no shadows
    #     shadow=False,
    #     # with colors
    #     colors=colors,
    #     # with one slide exploded out
    #     explode=(0.2, 0.1, 0.1),
    #     # with the start angle at 90%
    #     # with the percent listed as a fraction
    #     autopct='%1.1f%%',
    #     startangle=90
    # )
    #
    # for i in texts:
    #     i.set_fontsize(25)
    #     i.set_fontname("Laksaman")
    # for i in autotexts:
    #     i.set_fontsize(23)
    #
    # # View the plot drop above
    # ax1.axis('equal')
    # fig1.suptitle('Discrimination Classification', fontsize=30, fontname="Laksaman")
    #
    # fig1.savefig('media/pie.png', dpi=100)
    return pos,neg,neut





