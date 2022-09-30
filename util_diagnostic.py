"""
    Created by : Xarnact
    Date : 28-09-2022
    Qoutes : Are you 1 or 0 ?

"""
from locale import normalize
import pandas as pd
import numpy as np
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


    


def diagnosticCVResult():
    ...

