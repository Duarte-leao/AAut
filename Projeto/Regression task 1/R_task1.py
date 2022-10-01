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
X_test = np.load('Xtest_Regression1.npy')


# Testing multiple predictors with k-fold cross validation

k = 10 # Number of splits in cross validation

#### Ordinary Least Squares ####

lin_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

#### Ridge Regression ####

# Generate array of alphas
ridge_alphas = []
alpha = 1

for i in range(200):
    alpha = alpha/1.05
    ridge_alphas.append(alpha)

ridge_reg = linear_model.RidgeCV(alphas = ridge_alphas, scoring = 'neg_mean_squared_error', cv = k).fit(X, Y)
ridge_alpha = ridge_reg.alpha_ # Choose alpha that best fits the data
ridge_reg_MSE = abs(ridge_reg.best_score_) # Score of the best alpha

#### Lasso Regression ####

# Generate array of alphas
lasso_alphas = []
alpha = 1

for i in range(200):
    alpha = alpha/1.05
    lasso_alphas.append(alpha)

lasso_reg = linear_model.LassoCV(alphas = lasso_alphas, random_state = 0, cv = k).fit(X, np.ravel(Y))
lasso_alpha = lasso_reg.alpha_ # Choose alpha that best fits the data

lasso = linear_model.Lasso(alpha = lasso_alpha)
lasso_reg_scores = cross_validate(linear_model.Lasso(alpha = lasso_alpha), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
lasso_reg_MSE = abs(lasso_reg_scores['test_score'].mean())


#### Selection of best estimator to do prediction ####

predictors = [linear_model.LinearRegression(), linear_model.Ridge(alpha = ridge_alpha), linear_model.Lasso(alpha = lasso_alpha)]
MSE_array = np.array([lin_reg_MSE, ridge_reg_MSE, lasso_reg_MSE])
min_MSE = np.argmin(MSE_array)
# print(MSE_array)

predictor = predictors[min_MSE]

# Making the prediction
train_predictor = predictor.fit(X, Y)
y_predict = train_predictor.predict(X_test)
np.save('Y_Predicted.npy', y_predict)