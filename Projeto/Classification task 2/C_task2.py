"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 30/10/2022
Descrição: Classificação Parte 2
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score as BACC
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


X = np.load('Xtrain_Classification2.npy')
Y = np.load('Ytrain_Classification2.npy')
X_test = np.load('Xtest_Classification2.npy')

# print(np.shape(X))
# print(np.shape(Y))
# print(np.shape(X_test))

# print(X)
# print(Y)
# print(X_test)
 
# Images = 676= 26*26 patches
# MLP
# DT
# SVM
# random forest
# KNN
# Logistic regression

# 0 - white center
# 1 - rings
# 2 - background

#Normalize data   

X = X/255
X_test = X_test/255

Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)



# print(np.count_nonzero(Y==0))
# print(np.count_nonzero(Y==1))
# print(np.count_nonzero(Y==2))

def sampling_method(method, random_state = None):
    if method == 0:
        balancing_method = SMOTE(random_state= random_state)
    elif method == 1:
        balancing_method = RandomOverSampler(random_state= random_state)

    return balancing_method

def balance_data(x, y, method):

    sampling = sampling_method(method,1)
    X, Y = sampling.fit_resample(x, y)

    return X, Y

def hyperparameters_tunning(x_training, y_training, balance):
     
    scoring = {'score': met.make_scorer(BACC)}

    if balance == 1:
        parameters={ 'classifier__hidden_layer_sizes':[(128,64), (128,), (128,128)],  'classifier__alpha':[  0.001, 0.0005], 'classifier__learning_rate_init':[ 0.001]}
        pipeline = imbpipeline(steps = [['sampling', sampling_method(0,1)],['classifier', MLPClassifier()]])
        gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 5,refit="score", n_jobs=-1, verbose=2)
        gs_cv.fit(x_training,y_training)
        print(gs_cv.best_params_)
        print(gs_cv.best_score_)
    else :
        parameters={ 'hidden_layer_sizes':[(6,),(6,2), (10,), (14,)],  'alpha':[  0.001, 0.0005], 'learning_rate_init':[ 0.003,0.002, 0.001]}
        model = MLPClassifier()
        gs_cv1 = GridSearchCV(model , parameters, scoring=scoring, cv= 5,refit="score", verbose=2)
        gs_cv1.fit(x_training,y_training)
        print(gs_cv1.best_params_)
        print(gs_cv.best_score_)

# def hyperparameters_tunning(x_training, y_training, balance):
    
#     scoring = {'score': met.make_scorer(BACC)}

#     if balance == 1:
#         parameters={ 'classifier__n_neighbors':[3,4,5,10,20,30,50,],  'classifier__weights':['distance','uniform'], 'classifier__algorithm':['auto']}
#         pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', KNeighborsClassifier())])
#         gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 5,refit="score",n_jobs=-1, verbose=2)
#         gs_cv.fit(x_training,y_training)
#         print(gs_cv.best_params_)
#         print(gs_cv.best_score_)

    
hyperparameters_tunning(Xtrain,ytrain,1)


Xtrain, ytrain = balance_data(Xtrain, ytrain, 0)

# model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(14,),activation = 'relu', learning_rate_init=0.001)
# model = KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights= 'distance', n_jobs=-1)
# model.fit(Xtrain, ytrain)

# y_pred = model.predict(Xval)
# print(BACC(yval, y_pred))

# {'classifier__alpha': 0.001, 'classifier__hidden_layer_sizes': (14,), 'classifier__learning_rate_init': 0.001} MLP

# {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 4, 'classifier__weights': 'distance'} KNN

