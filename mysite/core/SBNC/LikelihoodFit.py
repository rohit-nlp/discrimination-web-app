
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.rinterface import RRuntimeError

def fit(df,adjacencyMatrix):


    categoricalMatrix = np.full(df.shape,"miss")
    dfMatrix = df.values

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if dfMatrix[i,j] == 1:
                categoricalMatrix[i,j] = "hit"

    categoricalMatrixDF = pd.DataFrame(categoricalMatrix, columns=df.columns)

    parent = list()
    child = list()
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(adjacencyMatrix.shape[1]):
            if i != j:
                if adjacencyMatrix[i, j] == 0:
                    parent.append(df.columns[i])
                    child.append(df.columns[j])

    #If not, bnlearn says dataset contains Na/NaN
    categoricalMatrixDF = categoricalMatrixDF.astype('category')

    blacklist = pd.DataFrame({'From': parent, 'To': child})
    #   blacklist.to_csv("blacklist.csv",index=None)


    ###R interface
    import rpy2.robjects.packages as rpackages

    # import R's utility package
    utils = rpackages.importr('utils')

    try:
        bnlearn = robjects.packages.importr('bnlearn')
    except RRuntimeError:
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages("bnlearn")
        bnlearn = robjects.packages.importr('bnlearn')

    pandas2ri.activate()

    #Conversion to R datasets
    #blacklist.to_csv("blacklist",sep=";",index=False)
    #categoricalMatrixDF.to_csv("matrix",sep=";",index=False)
    blacklist_to_R = pandas2ri.py2ri(blacklist)
    df_to_R = pandas2ri.py2ri(categoricalMatrixDF)

    training = bnlearn.hc(df_to_R,score="bic",blacklist=blacklist_to_R)
    #arcs is a matrix with 2 columns (from, to) with the computed arcs
    arcs = training[2]

    #Dic to map column names with index. Remember we did the blacklist with var names instead of numbers.
    indexColumns = {df.columns[i]: i for i in range(0, len(df.columns))}

    reconstructed = np.zeros(adjacencyMatrix.shape)
    for i in range(arcs.nrow):
        #arcs.rx(x,y) way to adress an R matrix. R matrix start with index 1 (lame)
        #asarray() used to convert a matrix position into a numpy array
        #asarray()[0] because the node name is located on position 0 of the array
        #indexColumns[] map the node name to node position
        #final result is just reconstructed[52,62]
        #we put a '1' becuase i->j means causation
        reconstructed[indexColumns[(np.asarray(arcs.rx(i + 1, 1))[0])], indexColumns[(np.asarray(arcs.rx(i + 1, 2))[0])]] = 1
    return reconstructed





    



