import igraph as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def walk(graph,start,end):
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
            #graph.es[edge index] for getting the edge, .es()["edgeprob"] to get his prob
            #graph_get_eid(Node A, Node B) to get the edge that goes from A to B.
            #graph.vs[vertex index] to get the neigh vertex, ["name"] to get his name and then convert it to 'Node B'
            probs.append(graph.es[graph.get_eid(node["name"], graph.vs[neighbor]["name"])]["edgeprob"])

        # Make probs sum up to 1 (needed for choice method)
        probs = [probs[i] / sum(probs) for i in range(len(probs))]
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
    posCount = 0
    negCount = 0
    neutralCount = 0
    for i in range(nIter):
        countPos = walk(graph,node,posName)[-1]
        countNeg = walk(graph,node,negName)[-1]
        if countPos == posName:
            #print("pos")
            posCount += 1
        elif countNeg == negName:
            #print("neg")
            negCount += 1
        else:
            neutralCount += 1
    return posCount,negCount,neutralCount



#This method creates our graph, computes the negative discrimination score for each node
#And returns it as a dict {nodeName:score}
def performRandomWalk(df,probs,nIter,posName,negName):
    negScores = list()
    posScores = list()
    var = list()
    neutralScores = list()
    #For each Node
    for i in range(df.shape[1]):
        #Get times we arrived in a negative and positive decision
        posScore,negScore,neutralScore = count(createGraph(probs),nIter,df.columns[i],posName,negName)
        #Caution, we cannot divide by zero
        if posScore + negScore ==0:
            negScores.append(0)
            posScores.append(0)
        else:
            #Negative Score = times we arrived negative decision / sum(negative+positive)
            negScores.append(negScore/nIter)
            posScores.append(posScore /nIter)
        neutralScores.append(neutralScore/nIter)
        var.append(df.columns[i])
    #Scores as dict then this dict sort it by value
    scores = pd.DataFrame({'Name':var,'Positive Score':posScores,'Negative Score':negScores,'Neutral Score':neutralScores})
    makePie(scores)
    return scores

def makePie(scores):

    scores = scores.drop(["Name"],axis=1)
    scores['max_value'] = scores.idxmax(axis=1)

    colors = ["#ffb39c", "#E3D4AD", "#6183A6"]
    labels = ["Negative", "Neutral", "Positive"]
    # Create a pie chart
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    patches, texts, autotexts = ax1.pie(
        # using data total)arrests
        scores.groupby('max_value').size(),
        # with the labels being officer names
        labels=labels,
        # with no shadows
        shadow=False,
        # with colors
        colors=colors,
        # with one slide exploded out
        explode=(0.2, 0.1, 0.1),
        # with the start angle at 90%
        startangle=90,
        # with the percent listed as a fraction
        autopct='%1.1f%%',
    )

    for i in texts:
        i.set_fontsize(25)
        i.set_fontname("Laksaman")
    for i in autotexts:
        i.set_fontsize(23)

    # View the plot drop above
    ax1.axis('equal')
    fig1.suptitle('Discrimination Classification', fontsize=30, fontname="Laksaman")

    fig1.savefig('media/pie.png', dpi=100)





