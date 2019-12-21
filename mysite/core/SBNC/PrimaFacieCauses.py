import pandas as pd
import numpy as np

def getPrimaFacieCauses(df, marginalProbs, primaFacieModel, primaFacieModelNotHappened, temporalOrder):

    adjacencyMatrix = np.full(((np.shape(primaFacieModel)[1]),np.shape(primaFacieModel)[1]),1).astype(float)
    np.fill_diagonal(adjacencyMatrix,0)

    #colnames?
    adjMatrixWithColNames = pd.DataFrame(adjacencyMatrix,None,df.columns)


    for i in range(np.shape(adjacencyMatrix)[0]):
        for j in range(i,np.shape(adjacencyMatrix)[1]):
            if i!=j:

                # Verify temporal priority
                atributeNameI = adjMatrixWithColNames.columns[i]
                atributeNameJ = adjMatrixWithColNames.columns[j]


                levelI = temporalOrder[temporalOrder['atribute'] == atributeNameI].iloc[0]['order']
                levelJ = temporalOrder[temporalOrder['atribute'] == atributeNameJ].iloc[0]['order']

                if levelI > 0 and levelJ > 0:
                    if levelI > levelJ: adjacencyMatrix[i,j] = 0
                    if levelJ > levelI: adjacencyMatrix[j,i] = 0
                else:
                    print("Err in TemporalOrder Dataframe")

    # Verify Probability Raising
    for i in range(np.shape(adjacencyMatrix)[0]):
        for j in range(i,np.shape(adjacencyMatrix)[1]):
            if i!=j:
                #P(j|i)>P(j|not i) the edge i --> j is valid for temporal priority
                if primaFacieModel[i,j] <= primaFacieModelNotHappened[i, j]: adjacencyMatrix[i,j] = 0
                if primaFacieModel[j,i] <= primaFacieModelNotHappened[j,i]: adjacencyMatrix[j,i] = 0

    return adjacencyMatrix


