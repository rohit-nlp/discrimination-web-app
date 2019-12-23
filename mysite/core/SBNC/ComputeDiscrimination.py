
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.rinterface import RRuntimeError

def pageRank(df,probs,posName,negName):
    posValues = list()
    negValues = list()
    datasetValues = df.values

    #Change the P(e|c)-P(e|not c) colum to an easy name: edgeprob
    #The probs df will always have 3 columns where the last one is P(e|c)-P(e|not c).
    #probs.columns = [probs.columns[0],probs.columns[1],"edgeprob"]

    ###R interface
    import rpy2.robjects.packages as rpackages

    # import R's utility package
    utils = rpackages.importr('utils')

    #Sees if you have already this packages installed or not
    try:
        igraph = robjects.packages.importr('igraph')
    except RRuntimeError:
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages("igraph")
        igraph = robjects.packages.importr('igraph')

    #To convert pandas to R
    pandas2ri.activate()
    probs_to_R = pandas2ri.py2ri(probs)


    #Create the positive and negative discrimination column names to R workspace
    robjects.globalenv['pos'] = posName
    robjects.globalenv['neg'] = negName

    #Create igraph
    graph = igraph.graph_from_data_frame(probs_to_R, directed=True)
    probs4Score = probs["edgeprob"]

    for i in range(df.shape[0]):
        #PageRank method, personalized for each individual in the dataset
        score = igraph.page_rank(graph, weights=probs4Score, personalized=datasetValues[i] / 10, directed=True, algo="prpack")
        #To R workspace so we can extract by name
        robjects.globalenv['score'] = score[0] #[0] is where the vector is stored
        posValues.append(robjects.r("score[pos]")[0]) #The [0] its because its contained in an array size 1
        negValues.append(robjects.r("score[neg]")[0])

    scores = pd.DataFrame({"Positive Discrimination":posValues,"Negative Discrimination":negValues})

    scores[df.columns] = df
    #Compute Generalized Discrimination Score


    gdsScore = gds(scores,"Positive Discrimination","Negative Discrimination")
    scores[gdsScore.columns] = gdsScore

    if (scores.shape[0] > 1) and (scores.shape[1] > 0):
        #scores.to_csv("Discrimination Results.csv",sep = ";",index = None)
        return scores
    return None

def gds(results,pos,neg):
    results['GDS-'] = results[neg] / (results[neg]+results[pos])
    results["GDS+"] = 1-results['GDS-']
    return results[['GDS+','GDS-']]



