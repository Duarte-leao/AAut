"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 22/10/2022
Descrição: Classificação Parte 1
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
#import tensorflow_addons as tfa
#from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
#from imblearn.pipeline import Pipeline as imbpipeline
#from imblearn.keras import BalancedBatchGenerator
#from imblearn.keras import balanced_batch_generator
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import NearMiss


X = np.load('Xtrain_Classification1.npy')
Y = np.load('ytrain_Classification1.npy')


Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.20, random_state=2)


####  NORMALIZE DATA   ####

Xtrain = Xtrain/255
Xval = Xval/255
# Xtest = Xtest/255


##############Logistic Regression###################
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

param_grid = {'C': [0.04, 0.05], 'max_iter': [2700]}
grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose=3)
grid.fit(Xtrain,ytrain)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions= grid.predict(Xval)
print('Logistic Regression F1 Score')
print(f1_score(yval,grid_predictions))

#Melhor Resultado para os parametros de LogisticRegression
#{'C': 0.05, 'max_iter': 2700}
#LogisticRegression(C=0.05, max_iter=2700)
#Logistic Regression F1 Score
#0.5927272727272729



##############Support vector machine###################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
param_grid = { 'gamma':['scale',0.009],'C': [2, 3, 4]}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
grid.fit(Xtrain,ytrain)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions= grid.predict(Xval)
print('Support vector machine F1 Score')
print(f1_score(yval,grid_predictions))

#Melhor Resultado para os parametros de Support vector machine
#{'C': 3, 'gamma': 0.009}
#SVC(C=3, gamma=0.009)
#0.7450657894736842