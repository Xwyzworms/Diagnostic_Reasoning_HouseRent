import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf
import pandas as pd

def doStratifiedCrossvalidation(model,x,y,n_splits=5, shuffle=True, random_state=42, isMLP = False, batch = 32,epochs=10):
    skf=sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    acc=[]
    recall=[]
    precision=[]
    f1=[]

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        if(isMLP):
            model.fit(X_train,Y_train, batch_size=batch, epochs=epochs)
        else :
            model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        if(isMLP):
            ...
        else:
            acc.append(sklearn.metrics.accuracy_score(Y_test, Y_pred))
            recall.append(sklearn.metrics.recall_score(Y_test, Y_pred,average='micro'))
            precision.append(sklearn.metrics.precision_score(Y_test, Y_pred,average='micro'))
            f1.append(sklearn.metrics.f1_score(Y_test, Y_pred,average='micro'))

            print(acc)
            print(recall)
            print(precision)
            print(f1)
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


def createMLP(default_neuron=100, dropoutparams=0.3, lr=0.01):
    model_l = tf.keras.Sequential()
    model_l.add(tf.keras.layers.Dense(default_neuron, input_shape=(10,)))
    model_l.add(tf.keras.layers.Dense(default_neuron*2))
    model_l.add(tf.keras.layers.Dropout(dropoutparams))
    model_l.add(tf.keras.layers.Dense(50))
    model_l.add(tf.keras.layers.Dense(5,activation='softmax' ))

    model_l.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "acc"
        ]
    )
    print(model_l.summary())
    return model_l



def doValidation():
    ...

"""
    Supervised Solution
"""
def prepareSVM(do_hyperParams=False):
    if(do_hyperParams):
        params = {
            "C" : (0.1, 1, 0.1)
        }
        model =sklearn.model_selection.GridSearchCV(SVC(), param_grid=params)
        return model
    return SVC()

def prepareKNN():
    return KNeighborsClassifier(n_neighbors=5,)

def prepareRandomForest(n_estimator=100, n_depth=3, do_hyperParams = False):
    if(do_hyperParams):
        parameters_RandomForest = {
        'n_estimators': (100,150, 50),
        'max_depth': (1,2,1),
        }
        model = sklearn.model_selection.GridSearchCV(RandomForestClassifier(), param_grid=parameters_RandomForest)
        return model
    return RandomForestClassifier(n_estimators= n_estimator, max_depth=n_depth)
    

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


