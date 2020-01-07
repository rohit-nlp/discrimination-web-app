
import pandas as pd
import igraph as p
import time

def pageRank(df,probs,posName,negName,varName):
    elapsed = time.time()
    datasetValues = df.values


    #Create the graph
    tuples = [tuple(x) for x in probs.values]
    graph = p.Graph.TupleList(tuples, directed=True, edge_attrs=['edgeprob'])

    #Pos and neg lists to store the score
    pos = list()
    neg = list()
    #Index of pos and neg nodes
    indexPos = -1
    indexNeg = -1

    #Get index of pos and neg nodes
    for i, j in enumerate(graph.vs):
        if j["name"] == posName:
            indexPos = i
        elif j["name"] == negName:
            indexNeg = i


    for i in range(df.shape[0]):
        scores = graph.personalized_pagerank(directed=True,weights='edgeprob',reset=datasetValues[i]/10,implementation = "prpack")
        pos.append(scores[indexPos])
        neg.append(scores[indexNeg])


    scores = pd.DataFrame({"Positive Discrimination":pos,"Negative Discrimination":neg})

    #scores[df.columns] = df
    #scores.to_csv("scores.csv",sep=";",index=None)
    scores[varName] = df[varName]
    #Compute Generalized Discrimination Score


    gdsScore = gds(scores,"Positive Discrimination","Negative Discrimination")
    scores[gdsScore.columns] = gdsScore
    elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
    if (scores.shape[0] > 1) and (scores.shape[1] > 0):
        return scores,elapsed
    return None

def gds(results,pos,neg):
    results['GDS-'] = results[neg] / (results[neg]+results[pos])
    results["GDS+"] = 1-results['GDS-']
    return results[['GDS+','GDS-']]



