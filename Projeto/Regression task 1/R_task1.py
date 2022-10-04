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

X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')
X_test = np.load('Xtest_Regression1.npy')


# Testing multiple predictors with k-fold cross validation

k = 10 # Number of splits in cross validation


Regressions_names = []
#### Ordinary Least Squares ####
Regressions_names.append('Ordinary Least Squares Regression')

lin_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

#### Ridge Regression ####
Regressions_names.append('Ridge Regression')
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
Regressions_names.append('Lasso Regression')
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

#### Elastic-Net Regression ####
Regressions_names.append('Elastic-Net Regression')
# Generate array of alphas and l1-ratios
elastic_alphas = []
elastic_l1_ratios = []
alpha = 1
l1_ratio = 1
for i in range(100):
    alpha = alpha/1.05
    elastic_alphas.append(alpha)
    if i < 75:
        l1_ratio = l1_ratio/1.007
        elastic_l1_ratios.append(l1_ratio)
    else:
        l1_ratio = l1_ratio/1.4
        elastic_l1_ratios.append(l1_ratio)

elastic_reg = linear_model.ElasticNetCV(alphas = elastic_alphas,l1_ratio=elastic_l1_ratios, random_state = 0, cv = k).fit(X, np.ravel(Y))
elastic_alpha = elastic_reg.alpha_ # Choose alpha that best fits the data
elastic_l1_ratio = elastic_reg.l1_ratio_ # Choose l1-ratio that best fits the data

elastic_net = linear_model.ElasticNet(alpha = lasso_alpha, l1_ratio = elastic_l1_ratio)
elastic_net_reg_scores = cross_validate(elastic_net, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
elastic_net_reg_MSE = abs(elastic_net_reg_scores['test_score'].mean())

#### Least Angle Regression ####
Regressions_names.append('Least Angle Regression')

LARS_reg_scores = cross_validate(linear_model.Lars(normalize = False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
LARS_reg_MSE = abs(LARS_reg_scores['test_score'].mean())


#### Lasso Least Angle Regression ####
Regressions_names.append('Lasso Least Angle Regression')

lassoLars_reg = linear_model.LassoLarsCV(cv = k, normalize=False).fit(X, np.ravel(Y))
lassoLars_alpha = lassoLars_reg.alpha_ # Choose alpha that best fits the data

lassoLars_reg_scores = cross_validate(linear_model.LassoLars(alpha = lassoLars_alpha, normalize=False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
lassoLars_reg_MSE = abs(lassoLars_reg_scores['test_score'].mean())

#### Orthogonal Matching Pursuit Regression ####
Regressions_names.append('Orthogonal Matching Pursuit Regression')

OMP_reg = linear_model.OrthogonalMatchingPursuitCV(cv = k, normalize=True, max_iter=10).fit(X, np.ravel(Y))
OMP_n_zero_coef = OMP_reg.n_nonzero_coefs_ # Choose number of non zero coefficients that best fits the data

OMP_reg_scores = cross_validate(linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = OMP_n_zero_coef, normalize=True), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
OMP_reg_MSE = abs(OMP_reg_scores['test_score'].mean())


#### Selection of best estimator to do prediction ####

predictors = [linear_model.LinearRegression(), linear_model.Ridge(alpha = ridge_alpha), linear_model.Lasso(alpha = lasso_alpha), linear_model.ElasticNet(alpha = lasso_alpha, l1_ratio = elastic_l1_ratio), linear_model.Lars(normalize=False), linear_model.LassoLars(alpha = lassoLars_alpha, normalize=False), linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = OMP_n_zero_coef, normalize=True)]
MSE_array = np.array([lin_reg_MSE, ridge_reg_MSE, lasso_reg_MSE, elastic_net_reg_MSE, LARS_reg_MSE, lassoLars_reg_MSE, OMP_reg_MSE])
min_MSE = np.argmin(MSE_array)

print(MSE_array)
print(min_MSE)
print('Best Predictor: ', Regressions_names[min_MSE])

predictor = predictors[min_MSE]

# Making the prediction
train_predictor = predictor.fit(X, Y)
y_predict = train_predictor.predict(X_test)
np.save('Y_Predicted.npy', y_predict)