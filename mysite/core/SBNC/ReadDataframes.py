#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import pandas as pd


# Function that given a path reads a .csv/.txt file and returns it as a Pandas dataframe
def read(pathDF, pathOrder):
    try:
        df = pd.read_csv(pathDF)
    except FileNotFoundError:
        print("Dataframe file not found!")
        return None, None
    try:
        temporalOrderDF = pd.read_csv(pathOrder)
    except FileNotFoundError:
        print("Temporal Order file not found!")
        return None, None
    return df, temporalOrderDF

def readOriginalDF(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Dataframe file not found!")
        return None, None
    return df