"""
    Created by : Xarnact
    Date : 28-09-2022
    Qoutes : Are you 1 or 0 ?

"""
import pandas as pd
import numpy as np

def getTotalUniqueColumns(df, columns, showColValue : bool = False):
    
    for column in columns:
        uniques = df[column].unique()
        print(f"Column : {column}, {len(uniques)}")
        if(showColValue):
            getShowUniqueInformation(uniques)
            print("\n")


def getShowUniqueInformation(arr : np.array):

    if(len(arr) > 10 ):
        print(arr[:10], arr.dtype)
    else:
        print(arr, arr.dtype)


def diagnosticCVResult():
    ...

