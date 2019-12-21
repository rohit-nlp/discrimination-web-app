import pandas as pd
import numpy as np

def read(pathDF,pathOrder):
    try:
        df = pd.read_csv(pathDF, sep=";")
    except FileNotFoundError:
        print("Dataframe file not found!")
        return None,None
    try:
        temporalOrderDF = pd.read_csv(pathOrder, sep=";")
    except FileNotFoundError:
        print("Temporal Order file not found!")
        return None,None
    return df,temporalOrderDF