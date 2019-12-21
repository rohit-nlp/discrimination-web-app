import pandas as pd
import numpy as np

from .DAGScores import DAG
from .PrimaFacieCauses import getPrimaFacieCauses

def getPrimaFacieParents(df,temporalOrder,jointProbs,marginalProbs):

    primaFacieModel,primaFacieModelNotHappened = DAG(df,jointProbs,marginalProbs)

    adjacencyMatrix = getPrimaFacieCauses(df,marginalProbs,primaFacieModel,primaFacieModelNotHappened,temporalOrder)

    return adjacencyMatrix

