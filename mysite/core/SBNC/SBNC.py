#setwd("C:/Users/blair/Desktop/reconstructions/GermanCreditBinary_formatted/SBCN")
#setwd("C:/Users/blai.ras/Desktop/Suppes-Bayes-Causal-Network-for-discrimination-detection/Algorithm/ProbCausalDisc/reconstructions/GermanCreditBinary_formatted/SBCN")

import time

from .ReadDataframes import read
from .DataframeCheck import dataframeCheck
from .PrimaFacieParents import getPrimaFacieParents
from .LikelihoodFit import fit
from .ComputeScores import score
from .ExportSBNC import export
from .ComputeDiscrimination import pageRank
from .WeightedRandomWalk import performRandomWalk
from .DataframeCheck import probCheck, temporalOrderCheck

def SBNC(pathDF,pathOrder,posColumn,negColumn):
    elapsed = time.time()
    df, temporalOrder = read(pathDF,pathOrder)
    #For Returns
    probs=None
    scoresDicts = None
    pos = None
    neg =None
    neut = None
    disconnectedNodes = None

    if df is not None:
        temporalOrder,reason = temporalOrderCheck(df, temporalOrder, posColumn, negColumn)
        if reason == "":
            # Check for NaN's, Nulls, Columns/Rows > 1
            reason = dataframeCheck(df)
            print("Dataframe is correct, starting probability computation")
            if reason == "":
                df, temporalOrder, marginalProbs, jointProbs, reason = probCheck(df,temporalOrder)
                #Prob checkight delete columns, so we have to check
                if reason == "":
                    adjacencyMatrix = getPrimaFacieParents(df,temporalOrder,jointProbs,marginalProbs)
                    print("adjacencyMatrix done")
                    adjMatrixReconstucted = fit(df,adjacencyMatrix)
                    print("Fit done")
                    nodes, probHappened,probNotHappened,substract = score(df,adjMatrixReconstucted,marginalProbs,jointProbs)
                    print("score done")
                    probs,df,disconnectedNodes = export(df, adjMatrixReconstucted, nodes,substract)
                    print("Export done")
                    #df = pd.read_csv("datasets/inputDataVector.csv",header=None)
                    #probs.to_csv("Probs.csv",sep=";",index=False)
                    if probs is not None:
                        print("SBNC Reconstruction finished with exit")
                        print("Starting discrimination scoring")
                        scoresDicts,pos,neg,neut = performRandomWalk(df,probs,1000,posColumn,negColumn)
                        #pageRank(df,probs,posColumn,negColumn)
                        elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
                    else:
                        reason = "After the reconstruction, the dataset has less than 2 columns"
    else:
        reason ="Error reading the file(s)"

    return reason,df,probs,scoresDicts,disconnectedNodes,pos,neg,neut,elapsed

def doPageRank(df,probs,posColumn,negColumn):
    return pageRank(df, probs, posColumn, negColumn)