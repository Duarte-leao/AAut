import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import TweedieRegressor
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


#### Generalized Linear Regression ####

####lin_reg_scores = cross_validate(linear_model.TweedieRegressor(power=0, alpha=0.5, link='log'), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
####lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

####print(lin_reg_MSE)

#### LARS Lasso (falta ver o alpha)####

####lin_reg_scores = cross_validate(linear_model.LassoLars(alpha=.001, normalize=False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
####lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

####print(lin_reg_MSE)

# Generate array of alphas
#lasso_alphas = []
#alpha = 1

#for i in range(200):
#    alpha = alpha/1.05
#    lasso_alphas.append(alpha)

lassoLars_reg = linear_model.LassoLarsCV(cv = k).fit(X, np.ravel(Y))
lassoLars_alpha = lassoLars_reg.alpha_ # Choose alpha that best fits the data

lassoLars = linear_model.LassoLars(alpha = lassoLars_alpha)
lassoLars_reg_scores = cross_validate(linear_model.LassoLars(alpha = lassoLars_alpha, normalize=True), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
lassoLars_reg_MSE = abs(lassoLars_reg_scores['test_score'].mean())

print(lassoLars_reg_MSE)