#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import pandas as pd
import numpy as np
from sklearn import preprocessing

# For checking if df is valid
def isAble(df, decisionCol):
    error = ""
    if df.isna().values.any() == True or df.isnull().values.any() == True:
        error = "Dataframe contains Null or NaN values"
    if df.shape[0] < 2 or df.shape[1] < 2:
        error = "Dataframe has less than 2 columns or less than 2 rows"
    if decisionCol not in df.columns:
        error = "Dataframe has not the decision column " + decisionCol
    return error


def getQuartiles(df, column):
    if column in df.columns:
        min = df[column].quantile(0)
        p25 = df[column].quantile(0.25)
        median = df[column].quantile(0.5)
        p75 = df[column].quantile(0.75)
        max = df[column].quantile(1)+1
        return min, p25, median, p75, max
    return None


def categorize4(df):
    newDf = pd.DataFrame()
    for column in df.columns:
        min,p25,median,p75,max =getQuartiles(df,column)
        bins = [min,p25,median,p75,max]

        if min == 0:
            colMin = "0"
        else: colMin = ('%f' % min).rstrip('.0')

        if min == p25:
            bins = [min, median, p75,max]
            names = [colMin +"_"+('%f' % median).rstrip('.0'),
                 ('%f' % median).rstrip('.0')+"_"+('%f' % p75).rstrip('.0'),
                 ('%f' % p75).rstrip('.0')+"_"+('%f' % max).rstrip('.0')]
        else:
            names = [colMin+"_"+('%f' % p25).rstrip('.0'),
                     ('%f' % p25).rstrip('.0')+"_"+('%f' % median).rstrip('.0'),
                     ('%f' % median).rstrip('.0')+"_"+('%f' % p75).rstrip('.0'),
                     ('%f' % p75).rstrip('.0')+"_"+('%f' % max).rstrip('.0')]
        tempDict = dict(enumerate(names, 1))
        newDf[column] = np.vectorize(tempDict.get)(np.digitize(df[column], bins))
    return newDf


def categorize2(df):
    newDf = pd.DataFrame()
    for column in df.columns:
        min, p25, median, p75, max = getQuartiles(df, column)
        bins = [min, median, max]
        if min == 0:
            colMin = "0"
        else: colMin = ('%f' % min).rstrip('.0')
        names = [colMin + "_" + ('%f' % median).rstrip('.0'), ('%f' % median).rstrip('.0') + "_" + ('%f' % max).rstrip('.0')]
        tempDict = dict(enumerate(names, 1))
        newDf[column] = np.vectorize(tempDict.get)(np.digitize(df[column], bins))
    return newDf


def dummyFeatures(df):
    # Deep copy the original data
    data_encoded = df.copy(deep=True)
    # Use Scikit-learn label encoding to encode character data
    lab_enc = preprocessing.LabelEncoder()
    for col in df.columns:
        data_encoded[col] = lab_enc.fit_transform(df[col])
        le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        print('Feature', col)
        print('mapping', le_name_mapping)

    # Create new dataframe with dummy features
    categorical_feats = df.select_dtypes(include=['object']).columns.tolist()
    finalDF = pd.get_dummies(df, columns=categorical_feats,prefix_sep="[")
    finalDF.columns = [i.replace(" ","_").replace("-","_")+"]" for i in finalDF.columns]
    return finalDF


def adaptDF(data, decisionCol):
    decisionCate = True
    posColumn = ""
    negColumn = ""
    df = data.copy(deep=True)
    df.columns = [i.replace("[","") for i in df.columns]
    error = isAble(df, decisionCol)

    if error == "":
        # Select categorical features
        categorical = df.select_dtypes(include=['object']).columns.tolist()
        numerical = [i for i in df.columns if i not in categorical]

        if decisionCol in numerical:
            numerical.remove(decisionCol)
            decisionCate = False

        # Transform numerical features
        dfNum = categorize4(df[numerical].copy(deep=True))

        # Transform  decision variable if its numerical
        if decisionCate == False:
            dfDecision = categorize2(df[[decisionCol]].copy(deep=True))
            dfDecision = dfDecision.join(dfNum)
            joinedDf = dfDecision.join(df[categorical])

        joinedDf = dfNum.join(df[categorical])


        posColumn, negColumn = decisionCol+"["+joinedDf[decisionCol].unique()+"]"

        # Transform categorical features (all of them)
        return dummyFeatures(joinedDf), posColumn, negColumn, error

    else:
        return None, posColumn, negColumn, error