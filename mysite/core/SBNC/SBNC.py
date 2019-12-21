#setwd("C:/Users/blair/Desktop/reconstructions/GermanCreditBinary_formatted/SBCN")
#setwd("C:/Users/blai.ras/Desktop/Suppes-Bayes-Causal-Network-for-discrimination-detection/Algorithm/ProbCausalDisc/reconstructions/GermanCreditBinary_formatted/SBCN")


from .ReadDataframes import read
from .DataframeCheck import dataframeCheck
from .PrimaFacieParents import getPrimaFacieParents
from .LikelihoodFit import fit
from .ComputeScores import score
from .ExportSBNC import export
from .ComputeDiscrimination import pageRank
from .WeightedRandomWalk import performRandomWalk
from .DataframeCheck import probCheck

def SBNC(pathDF,pathOrder,posColumn,negColumn):

    df, temporalOrder = read(pathDF,pathOrder)

    if df is not None:
        #Check for NaN's, Nulls, Columns/Rows > 1
        reason = dataframeCheck(df,temporalOrder)
        if reason == "":
            print("Dataframe is correct, starting probability computation")

            df, temporalOrder, marginalProbs, jointProbs, reason = probCheck(df,temporalOrder)

            #Prob checkight delete columns, so we have to check
            if reason == "":
                adjacencyMatrix = getPrimaFacieParents(df,temporalOrder,jointProbs,marginalProbs)
                adjMatrixReconstucted = fit(df,adjacencyMatrix)
                nodes, probHappened,probNotHappened,substract = score(df,adjMatrixReconstucted,marginalProbs,jointProbs)

                probs,df = export(df, adjMatrixReconstucted, nodes,substract)
                #df = pd.read_csv("datasets/inputDataVector.csv",header=None)
                if probs is not None:
                    print("SBNC Reconstruction finished with exit")
                    print("Starting discrimination scoring")
                    scoreDic = performRandomWalk(df,probs,1000,"positive_dec","negative_dec")
                    print(scoreDic)
                    pageRankScores = pageRank(df,probs,posColumn,negColumn)

                    if pageRankScores is not None:
                        print("Discrimination computed successfully")
                    else:
                        print("Something went wrong evaluating the discrimination score")
                else:
                    print("Something went wrong")
            else:
                print(reason)
        else:
            print(reason)
    else:
        print("Error reading the file(s)")
