"""
    Created by : Xarnact
    Date : 28-09-2022
    Qoutes : Are you 1 or 0 ?

"""
from locale import normalize
import pandas as pd
import numpy as np
import util_calculation

from scipy.stats import chi2_contingency
from scipy import stats
from IPython.display import display


def getTotalUniqueColumns(df, columns, showColValue : bool = False):
    
    for column in columns:
        uniques = df[column].unique()
        print(f"Column : {column}, unique: {len(uniques)}, dtype: {uniques.dtype}")
        if(showColValue):
            pd.set_option('display.max_rows', len(df[column]))
            if df[column].dtype == 'object':
                _getUniqueCountPercentageObj(df, column)
            else:
                display(df[column].describe())
            pd.reset_option('display.max_rows')

def _getUniqueCountPercentageObj(df, column):
    data=pd.DataFrame(df[column].value_counts(normalize=False))
    percentage=pd.DataFrame(df[column].value_counts(normalize=True))
    data['percentage']=percentage
    display(data) 

def getNumericColumns():
    # Buat Manual;
    ...

def getCategoricalColumns():
    # Buat Manual
    ...

def getAllPartialCorrelation(df):
    return df.pcorr()


def getChiSquareResult(chiList):
    p, chi = [], [] 
    indexCols = []
    columns = []

    for chiData in chiList:
        if(chiData[0] not in indexCols):
            indexCols.append(chiData[0])

        if(chiData[1] not in columns):
            columns.append(chiData[1])
        p.append(round(chiData[2][1],2))
        chi.append(round(chiData[2][0],2))
    pDf = pd.DataFrame(columns=columns)
    chiDf = pd.DataFrame(columns=columns)
    for data in range(0, len(chiList) , 10):
        p_information = p[data:data+10]
        chi_information = chi[data:data+10]
        pDf.loc[len(pDf)] = p_information
        chiDf.loc[len(chiDf)] = chi_information
    
    pDf["cols"] = columns
    chiDf["cols"] = columns
    pDf.set_index("cols",inplace=True)
    chiDf.set_index("cols",inplace=True)

    return pDf,chiDf


def getChiSquareSummary(df, cols_params):
    ans = []
    columns = ["column1", "column2", "chi_square", "p_value"]
    returned_df = pd.DataFrame(columns=columns)
    pairs1 = []
    for val1 in cols_params:
        for val2 in cols_params:
            if(isPairsExists((val1,val2),pairs1)):
                continue
            if(val1 == val2): 
                continue
            pairs1.append( (val1, val2))
            chiValue = util_calculation.calculateChiSquareIndependence(df,val1,val2)
            insertData = {
                columns[0] : val1,
                columns[1] : val2,
                columns[2] : chiValue[0],
                columns[3] : chiValue[1]
            }
            returned_df = returned_df.append(insertData, ignore_index=True)

    return returned_df


def getCramers_V(df,cols_param):
    ansd = []
    for var1 in cols_param:
        col = []
        for var2 in cols_param:
            cramer = util_calculation.calculateCramers_V(df, var1, var2)
            col.append(cramer)
        ansd.append(col)
    result = np.array(ansd)
    return pd.DataFrame(result, columns=cols_param,index=cols_param)



def getChiSquare(df, cols_params):
    ans = []
    for val1 in cols_params:
        for val2 in cols_params:
            relation = []
            relation.append(val1)
            relation.append(val2)

            chiValue = util_calculation.calculateChiSquareIndependence()

            relation.append(chiValue)

            ans.append(relation)
    return getChiSquareResult()


def getSumAndPercentageOfMissingValues(df):
    arr_sum_miss_val = df.isna().sum().values

    arr_perc_miss_val = (df.isna().sum() / len(df) * 100).values

    miss_val_df = pd.DataFrame(data={"Sum Missing Values" : arr_sum_miss_val, "Percentage Missing Values(%)" : arr_perc_miss_val}, index=df.columns)
    
    return miss_val_df

def diagnosticCVResult():
    ...


# def isPairsExists(pairs, pairs2):
#     for p in pairs2:
#     if( (p[0] == pairs[0] and p[1] == pairs[1]) or (p[1] == pairs[0] and p[0] == pairs[1] )):
#             return True
#     return False

