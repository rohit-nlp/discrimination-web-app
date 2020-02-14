#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import numpy as np
import pandas as pd
from sklearn import preprocessing


# For checking if df is valid for categoraritzation
def isAble(df, decisionCol):
    error = ""
    if df.isna().values.any() == True or df.isnull().values.any() == True:
        error = "Dataframe contains Null or NaN values"
    if df.shape[0] < 2 or df.shape[1] < 2:
        error = "Dataframe has less than 2 columns or less than 2 rows"
    if decisionCol not in df.columns:
        error = "Dataframe has not the decision column called '" + decisionCol + "'"
    return error


# Function that returns the minimum, maximum, first quartile, median and third quartile of a data sample
def getQuartiles(df, column):
    if column in df.columns:
        min = df[column].quantile(0)
        p25 = df[column].quantile(0.25)
        p75 = df[column].quantile(0.75)
        max = df[column].quantile(1) + 1
        return min, p25, p75, max
    return None


# Function that splits (categorizes) a numerical data sample in 3 bins by looking at its quartiles
def categorize3(df):
    newDf = pd.DataFrame()
    # For every column, get its quartiles
    for column in df.columns:
        min, p25, p75, max = getQuartiles(df, column)
        bins = [min, p25, p75, max]
        minString = str(min)
        p25String = str(p25)
        p75String = str(p75)
        maxString = str(max)

        names = ["Low","Average","High"]



        # # I want to remove the '.0' of floats when str(float) but if the float its (0.0) then I remove the whole string
        # if min == 0:
        #     colMin = "0"
        # else:
        #     colMin = minString.replace(".0", "")
        # # If the first quartile is also zero, I just need 3 splits
        # if min == p25:
        #     bins = [min, median, p75, max]
        #
        #     names = [colMin + "_" + medianString,
        #              medianString + "_" + p75String,
        #              p75String + "_" + maxString]
        # else:
        #     names = [colMin + "_" + p25String,
        #              p25String + "_" + medianString,
        #              medianString + "_" + p75String,
        #              p75String + "_" + maxString]




        # Enumerate & vectorize the numerical values
        tempDict = dict(enumerate(names, 1))
        newDf[column] = np.vectorize(tempDict.get)(np.digitize(df[column], bins))
    return newDf


# Takes a df and categorizes it with numerical and categorical variables in it.
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
    finalDF = pd.get_dummies(df, columns=categorical_feats, prefix_sep="[")
    finalDF.columns = [replaceAll(i) + "]" for i in finalDF.columns]
    return finalDF


# Our training library doesn't support some symbols, so I'm removing it
def replaceAll(string):
    invalidDict = {' ': "_", "-": "negative", ")": "", "(": "", "&": "_and_", "<": "minus", ">": "bigger",
                   "<=": "minus_or_equal", ">=": "bigger_or_equal",
                   "*": "", "'": "", '"': "", "=": "_equal_", "/": "_", '  ': ""}
    for i, j in invalidDict.items():
        string = string.replace(i, j)
    return string


# Function that manages the categorization of a dataframe
def adaptDF(data, decisionCol):
    posColumn = ""
    negColumn = ""
    df = data.copy(deep=True)
    # My separators are []
    df.columns = [i.replace("[", "") for i in df.columns]
    # Check if the df can be categorized
    error = isAble(df, decisionCol)
    if error == "":
        # Select categorical features
        categorical = df.select_dtypes(include=['object']).columns.tolist()
        numerical = [i for i in df.columns if i not in categorical]

        if decisionCol in numerical:
            error = "Decision Variable can't be numerical: how do I split the data?"
        else:
            # Get Decision Variable
            posColumn, negColumn = decisionCol + "[" + df[decisionCol].unique() + "]"
            posColumn = replaceAll(posColumn)
            negColumn = replaceAll(negColumn)
            # Check if I have to categorize numerical variables
            if len(numerical) > 0:
                # Transform numerical features
                dfNum = categorize3(df[numerical].copy(deep=True))
                joinedDf = dfNum.join(df[categorical])
                # Transform categorical features (all of them)
                return dummyFeatures(joinedDf), posColumn, negColumn, error
            else:
                return dummyFeatures(df), posColumn, negColumn, error

    return None, posColumn, negColumn, error
