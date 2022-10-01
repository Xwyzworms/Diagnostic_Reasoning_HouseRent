import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd

def doStratifiedCrossvalidation(model,x,y,n_splits=10, shuffle=True, random_state=42):
    skf=sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    acc=[]
    recall=[]
    precision=[]
    f1=[]

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        acc.append(sklearn.metrics.accuracy_score(Y_test, Y_pred))
        recall.append(sklearn.metrics.recall_score(Y_test, Y_pred))
        precision.append(sklearn.metrics.precision_score(Y_test, Y_pred))
        f1.append(sklearn.metrics.f1_score(Y_test, Y_pred))

    return np.array(acc).mean(), np.array(recall).mean(), np.array(precision).mean(), np.array(f1).mean()

def getStratifedKFoldScore(list_model,x,y,n_splits=10, shuffle=True, random_state=42):
    dict_score={
        'model_name':[],
        'accuracy':[],
        'recall':[],
        'precision':[],
        'f1':[]

    }
    for model in list_model:
        acc,recall,precision,f1=doStratifiedCrossvalidation(model,x,y,n_splits, shuffle, random_state)
        dict_score['model_name'].append(model.__class__.__name__)
        dict_score['accuracy'].append(acc)
        dict_score['recall'].append(recall)
        dict_score['precision'].append(precision)
        dict_score['f1'].append(f1)
        
    return pd.DataFrame(dict_score)



def splitDatasetAccordingly():
    ...

def doValidation():
    ...

"""
    Supervised Solution
"""
def prepareSVM():
    return SVC()

def prepareKNN():
    return KNeighborsClassifier()

def prepareRandomForest():
    ...

def prepareGradientBoosting():
    ...

def prepareXGBoost():
    ...

def doHyperparameterTuningTraditional():
    ...

def doHyperparameterTuningBoosting():
    ...


"""
Unsupervised Solution
"""
def createKModes():
    ...


