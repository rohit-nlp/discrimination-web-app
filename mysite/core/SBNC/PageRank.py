#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import igraph as p
import pandas as pd

# Function that computes the Personalized Page Rank score for every individual on the dataset
def pageRank(df, probs, posName, negName, varName):
    datasetValues = df.values  # This will be the personalizing vector
    N = df.shape[0]

    # Create the graph
    tuples = [tuple(x) for x in probs.values]
    graph = p.Graph.TupleList(tuples, directed=True, edge_attrs=['edgeprob'])

    # Pos and neg lists to store the score
    pos = list()
    neg = list()

    # Index of pos and neg nodes
    indexPos = -1
    indexNeg = -1

    # Get index of pos and neg nodes
    for i, j in enumerate(graph.vs):
        if j["name"] == posName:
            indexPos = i
        elif j["name"] == negName:
            indexNeg = i

    # For every individual:
    for i in range(df.shape[0]):
        scores = graph.personalized_pagerank(directed=True, weights='edgeprob', reset=datasetValues[i] / N,
                                             implementation="prpack")
        # Scores is a vector that sums 1 and every position is the score of a node. I only want pos and neg values:
        pos.append(scores[indexPos])
        neg.append(scores[indexNeg])

    scores = pd.DataFrame({"Positive Discrimination": pos, "Negative Discrimination": neg})

    # scores[df.columns] = df if you want all the attributes
    scores[varName] = df[varName]

    # Compute Generalized Discrimination Score (not used)
    gdsScore = gds(scores, "Positive Discrimination", "Negative Discrimination")
    scores[gdsScore.columns] = gdsScore

    # A regular check
    if (scores.shape[0] > 1) and (scores.shape[1] > 0):
        scores.to_csv("Scores.csv",index=None)
        return scores
    return None


def gds(results, pos, neg):
    results['GDS-'] = results[neg] / (results[neg] + results[pos])
    results["GDS+"] = 1 - results['GDS-']
    return results[['GDS+', 'GDS-']]
