"""
*******************************************************************************************

Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 1/10/2022
Descrição: Regressão Parte 1
              
*******************************************************************************************
"""
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from statistics import mean

# Load the data set

X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')
Xtest = np.load('Xtest_Regression1.npy')


# Testing multiple predictors with k-fold cross validation

k = 10 # Number of splits in cross validation

#### Ordinary Least Squares ####

lin_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

print(lin_reg_MSE)





