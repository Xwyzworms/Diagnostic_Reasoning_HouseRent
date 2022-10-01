import statsmodels
import sklearn
import scipy
import util_preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np
import pandas as pd
import scipy.stats as stats



def calculateIQR(data,axis=0):
    """
    It takes a dataframe and returns the interquartile range of all the numeric columns in the dataframe
    
    :param data: The dataframe
    :param axis: 0 for columns, 1 for rows, defaults to 0 (optional)
    :return: The interquartile range (IQR) is a measure of statistical dispersion, being equal to the
    difference between 75th and 25th percentiles, or between upper and lower quartiles, IQR = Q3 − Q1.
    In other words, the IQR is the first quartile subtracted from the third quartile; these quartiles
    can be clearly seen on a box plot on
    """
    col = util_preprocessing.getOnlyColmnsWithType(data, 'number')
    return scipy.stats.iqr(data[col],axis=axis)
    

def calculateCorrelationNumeric(data):
    """
    It takes a dataframe and returns a correlation matrix and a heatmap of the correlation matrix.
    
    :param data: The dataframe to be 
    
    Method of correlation:

    pearson : standard correlation coefficient (untuk numerik)

    kendall : Kendall Tau correlation coefficient (untuk nominal)

    spearman : Spearman rank correlation (untuk numerik)
    """

    profile=util_preprocessing.profileReport(data)
    _getCorellationFromProfillingReport(profile,"pearson")
    _getCorellationFromProfillingReport(profile,"kendall")
    _getCorellationFromProfillingReport(profile,"spearman")

def _getCorellationFromProfillingReport(profile,keys):
    """
    It takes a profile object and a list of keys and returns a correlation matrix and a heatmap 
    
    :param profile: the dataframe that you want to get the correlation matrix from
    :param keys: The name of the correlation matrix that you want to get
    """
    print("Correlation Matrix for "+keys)   
    matrix_cor=profile.description_set["correlations"][keys]
    display(matrix_cor)
    plt.title("Correlation Heatmap for "+keys)
    dataplot = sns.heatmap(matrix_cor, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    display(dataplot)
    plt.show()
        

def calculateCorrelationCategorical(data):
    profile=util_preprocessing.profileReport(data)
    _getCorellationFromProfillingReport(profile,"cramers")

def calculateChiSquareIndependence(df, val1, val2):
    crosstab = pd.crosstab(df[val1], df[val2], rownames= None, colnames=None)
    chiSquared = stats.chi2_contingency(crosstab)
    return chiSquared

def calculateCorrelationMix(data):
    profile=util_preprocessing.profileReport(data)
    _getCorellationFromProfillingReport(profile,"phi_k")
    

def calculateCramers_V(data, col1, col2):
    crosstab = np.array(pd.crosstab(data[col1], data[col2], rownames=None, colnames=None))
    chiSquare = stats.chi2_contingency(crosstab)
    
    observation = np.sum(crosstab)
    minimumValue = min(crosstab.shape) - 1
    print(minimumValue)
    return chiSquare / (observation * minimumValue)

def getOutlierValue(df, col, Q1, Q3, IQR):
    return ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

