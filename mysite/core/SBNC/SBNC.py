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
    explainable = None
    inco = None
    apparent = None
    disconnectedNodes = None
    scores = None
    invalidMarginal = None
    notDistinguish = None

    if df is not None:
        temporalOrder,reason = temporalOrderCheck(df, temporalOrder, posColumn, negColumn)
        if reason == "":
            # Check for NaN's, Nulls, Columns/Rows > 1
            reason = dataframeCheck(df)
            print("Dataframe is correct, starting probability computation")
            if reason == "":
                df, temporalOrder, invalidMarginal, notDistinguish, marginalProbs, jointProbs, reason = probCheck(df,temporalOrder)
                #Prob checkight delete columns, so we have to check
                if reason == "":
                    adjacencyMatrix = getPrimaFacieParents(df,temporalOrder,jointProbs,marginalProbs)
                    print("adjacencyMatrix done")
                    adjMatrixReconstucted = fit(df,adjacencyMatrix)
                    print("Fit done")
                    nodes, probHappened,probNotHappened,substract = score(df,adjMatrixReconstucted,marginalProbs,jointProbs)
                    print("score done")
                    probs,df,disconnectedNodes = export(df, adjMatrixReconstucted, nodes,substract)
                    #df.to_csv("DF After SBNC.csv",sep=";",index=None)
                    print("Export done")
                    #df = pd.read_csv("datasets/inputDataVector.csv",header=None)
                    #probs.to_csv("Probs.csv",sep=";",index=False)
                    if probs is not None:
                        print("SBNC Reconstruction finished with exit")
                        print("Starting discrimination scoring")
                        #probs.to_csv("probs.csv",sep=";",index=None)
                        scores,pos,neg,neut,explainable,inco,apparent = performRandomWalk(df,probs,1000,posColumn,negColumn,0.55,0.25)
                        #scores.to_csv("scorees.csv",index=None)
                        #pageRank(df,probs,posColumn,negColumn)
                        elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
                    else:
                        reason = "After the reconstruction, the dataset has less than 2 columns"
    else:
        reason ="Error reading the file(s)"

    return reason,df, invalidMarginal, notDistinguish,probs,scores,disconnectedNodes,pos,neg,neut,explainable,inco,apparent,elapsed

def doPageRank(df,probs,posColumn,negColumn):
    return pageRank(df, probs, posColumn, negColumn)