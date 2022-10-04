"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 1/10/2022
Descrição: Regressão Parte 1
*******************************************************************************************
"""
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from statistics import mean

# Load the data set

X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')


