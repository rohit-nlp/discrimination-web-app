import igraph as p
import pandas as pd
import numpy as np

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
    for i in range(nIter):
        toCount = walk(graph,node,posName)[-1]
        if toCount == posName:
            posCount += 1
        if toCount == negName:
            negCount += 1

    return posCount,negCount

#This method creates our graph, computes the negative discrimination score for each node
#And returns it as a dict {nodeName:score}
def performRandomWalk(df,probs,nIter,posName,negName):
    scores = list()
    #For each Node
    for i in range(df.shape[1]):
        #Get times we arrived in a negative and positive decision
        posScore,negScore = count(createGraph(probs),nIter,df.columns[i],posName,negName)
        #Caution, we cannot divide by zero
        if posScore + negScore ==0:
            scores.append((df.columns[i],0))
        else:
            #Negative Score = times we arrived negative decision / sum(negative+positive)
            scores.append((df.columns[i],negScore/(posScore+negScore)))
    #Scores as dict then this dict sort it by value
    return {k: v for k, v in sorted(dict(scores).items(), key=lambda item: item[1])}


