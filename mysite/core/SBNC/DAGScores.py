import pandas as pd
import numpy as np

def DAG(df,jointProbs,marginalProbs):

    primaFacieModel = np.full((df.shape[1], df.shape[1]), 0).astype(float)
    primaFacieModelNotHappened = np.full((df.shape[1], df.shape[1]), 0).astype(float)

    for i in range(np.shape(primaFacieModel)[0]):
        for j in range(i,np.shape(primaFacieModel)[1]):
            if (i != j):
                if marginalProbs[i] > 0 and marginalProbs[i] < 1 and marginalProbs[j] > 0 and marginalProbs[j] < 1:
                    if jointProbs[i, j] / marginalProbs[j] < 1 or jointProbs[i, j] / marginalProbs[i] < 1:
                        primaFacieModel[i, j] = jointProbs[j, i] / marginalProbs[i]
                        primaFacieModelNotHappened[i, j] = (marginalProbs[j] - jointProbs[j, i]) / (1 - marginalProbs[i])
                        primaFacieModel[j, i] = jointProbs[i, j] / marginalProbs[j]
                        primaFacieModelNotHappened[j, i] = (marginalProbs[i] - jointProbs[i, j]) / (1 - marginalProbs[j])
            else:
                primaFacieModel[i, j] = 1
                primaFacieModelNotHappened[i, j] = 0

    return primaFacieModel, primaFacieModelNotHappened
