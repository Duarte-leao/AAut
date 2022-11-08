"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 1/10/2022
Descrição: Regressão Parte 2
*******************************************************************************************
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import ensemble
from sklearn import neighbors
from sklearn import preprocessing
from statistics import mean
import matplotlib.pyplot as plt

# Load the data set

X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')
Regressions_names = []
Predictors = []
MSE_array = np.array([])

###################################################### Robust regressors to outliers ######################################################
def Robust_regressors(X,Y,Regressions_names,MSE_array,Predictors):
    # Data normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    

    # Testing multiple predictors with k-fold cross validation

    k = 30 # Number of splits in cross validation

    ##### KNN Regressor #####

    KNN_reg = neighbors.KNeighborsRegressor(n_neighbors=35,weights='distance',algorithm='brute')
    KNN_reg_scores = cross_validate(KNN_reg, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    KNN_reg_MSE = abs(KNN_reg_scores['test_score'].mean())
    # print('KNN MSE:', KNN_reg_MSE)
    Regressions_names.append('KNN Regressor')
    Predictors.append(KNN_reg)
    MSE_array = np.append(MSE_array,KNN_reg_MSE)

    ##### Huber Regressor #####

    Huber_reg = linear_model.HuberRegressor(max_iter=500, epsilon=20)
    Huber_reg_scores = cross_validate(Huber_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    Huber_reg_MSE = abs(Huber_reg_scores['test_score'].mean())
    # print('Huber Regressor MSE:', Huber_reg_MSE)
    Regressions_names.append('Huber Regressor')
    Predictors.append(Huber_reg)
    MSE_array = np.append(MSE_array,Huber_reg_MSE)

    ##### RANSAC Regressor #####

    RANSAC_reg = linear_model.RANSACRegressor(min_samples=20,residual_threshold=1,loss='squared_error')
    RANSAC_reg_scores = cross_validate(RANSAC_reg, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    RANSAC_reg_MSE = abs(RANSAC_reg_scores['test_score'].mean())
    # print('RANSAC Regressor MSE:', RANSAC_reg_MSE)
    Regressions_names.append('RANSAC Regressor')
    Predictors.append(RANSAC_reg)
    MSE_array = np.append(MSE_array,RANSAC_reg_MSE)

    ##### TheilSen Regressor #####

    TheilSen_reg = linear_model.TheilSenRegressor(n_subsamples=20)
    TheilSen_reg_scores = cross_validate(TheilSen_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    TheilSen_reg_MSE = abs(TheilSen_reg_scores['test_score'].mean())
    # print('TheilSen Regressor MSE:', TheilSen_reg_MSE)
    Regressions_names.append('TheilSen Regressor')
    Predictors.append(TheilSen_reg)
    MSE_array = np.append(MSE_array,TheilSen_reg_MSE)

    # ##### Random Forest Regressor #####

    RdmFor_reg = ensemble.RandomForestRegressor()
    RdmFor_reg_scores = cross_validate(RdmFor_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    RdmFor_reg_MSE = abs(RdmFor_reg_scores['test_score'].mean())
    # print('Random Forest MSE:', RdmFor_reg_MSE)
    Regressions_names.append('Random Forest Regressor')
    Predictors.append(RdmFor_reg)
    MSE_array = np.append(MSE_array,RdmFor_reg_MSE)

    return Regressions_names, Predictors, MSE_array

# Regressions_names, Predictors, MSE_array = Robust_regressors(X,Y,Regressions_names,MSE_array,Predictors)


###################################################### Outlier removal ######################################################
Outliers_methods_X_Y = {'X':{},'Y':{}}
Outliers_methods_errors = np.array([])
Outliers_methods_names = []
##### IQR #####
def outlier_removal_IQR(X,Y):
    df_raw_x=pd.DataFrame(X)
    df_raw_y=pd.DataFrame(Y)

    def outliers_y(df, ft):
        Q1=df[ft].quantile(0.25)
        Q3=df[ft].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        ls_y = df.index[(df[ft] < lower_bound) | (df[ft]>upper_bound)] 
        
        return ls_y

    def outliers_x(df, ft):
        Q1=df[ft].quantile(0.25)
        Q3=df[ft].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        ls_x = df.index[(df[ft] < lower_bound) | (df[ft]>upper_bound)] 
        
        return ls_x

    index_list_y=[]
    index_list_x=[]

    for index in range(len(df_raw_y.columns)):
        index_list_y.extend(outliers_y(df_raw_y,index))

    for index in range(len(df_raw_x.columns)):
        index_list_x.extend(outliers_x(df_raw_x,index))
      
    def remove(df,ls):
        df=df.drop(ls)
        return df

    index_list=[]

    for i in range(len(index_list_x)):
        if index_list_x[i] not in index_list:
            index_list.append(index_list_x[i]) 

    for j in range(len(index_list_y)):
        if index_list_y[j] not in index_list:
            index_list.append(index_list_y[j])
        

    df_x_cleaned= remove(df_raw_x,index_list)
    df_y_cleaned= remove(df_raw_y,index_list)

    X = df_x_cleaned.to_numpy()
    Y = df_y_cleaned.to_numpy()

    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)

  
    linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean())

    return X, Y, abs(linear_reg_scores['test_score']).mean()
Outliers_methods_errors = np.append(Outliers_methods_errors,outlier_removal_IQR(X,Y)[2])
Outliers_methods_names.append('IQR')
Outliers_methods_X_Y['X']['IQR'] = outlier_removal_IQR(X,Y)[0]
Outliers_methods_X_Y['Y']['IQR'] = outlier_removal_IQR(X,Y)[1]


##### Outlier removal via 100-fold cross validation #####
def outlier_removal_LOO(X, Y):
    while True:
        linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = X.shape[0], scoring = 'neg_mean_squared_error', return_train_score = True)
        linear_reg_MSE = abs(linear_reg_scores['test_score'])
        if linear_reg_MSE.max() > 1:
            # print('Outlier MSE:',linear_reg_MSE.max())
            X=np.delete(X,np.argmax(linear_reg_MSE),0)
            Y=np.delete(Y,np.argmax(linear_reg_MSE),0)
        else:
            break
    
    linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean())  


    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)

    return X, Y, abs(linear_reg_scores['test_score']).mean()
Outliers_methods_errors = np.append(Outliers_methods_errors,outlier_removal_LOO(X,Y)[2])
Outliers_methods_names.append('LOO')
Outliers_methods_X_Y['X']['LOO'] = outlier_removal_LOO(X,Y)[0]
Outliers_methods_X_Y['Y']['LOO'] = outlier_removal_LOO(X,Y)[1]

##### Local Outlier Factor #####
def LOF(X,Y):
    errors = np.array([])
    for i in range(1,100): # loop to choose the n_neighbors that help best with outlier removal 
        lof = LocalOutlierFactor( n_neighbors=i,contamination = 0.2)
        outlier_pred = lof.fit_predict(Y)
        mask = outlier_pred != -1
        X_cl, Y_cl = X[mask, :], Y[mask]
        linear_reg_scores = cross_validate(linear_model.LinearRegression(), X_cl, Y_cl, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
        errors = np.append(errors,abs(linear_reg_scores['test_score']).mean())

    lof = LocalOutlierFactor( n_neighbors=errors.argmin()+1,contamination = 0.2)
    outlier_pred = lof.fit_predict(Y)
    mask = outlier_pred != -1
    X, Y = X[mask, :], Y[mask]
    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)
    linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean())

    return X, Y, abs(linear_reg_scores['test_score']).mean()
Outliers_methods_errors = np.append(Outliers_methods_errors,LOF(X,Y)[2])
Outliers_methods_names.append('LOF')
Outliers_methods_X_Y['X']['LOF'] = LOF(X,Y)[0]
Outliers_methods_X_Y['Y']['LOF'] = LOF(X,Y)[1]


##### Elliptic Envelope #####

def Ellipt_Envelop(X, Y):
    Ellip_Env = EllipticEnvelope(contamination = 0.2)
    outlier_pred = Ellip_Env.fit_predict(Y)

    mask = outlier_pred != -1
    X, Y = X[mask, :], Y[mask]
    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)
    linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean()) 
    return X, Y, abs(linear_reg_scores['test_score']).mean()


Outliers_methods_errors = np.append(Outliers_methods_errors,Ellipt_Envelop(X,Y)[2])
Outliers_methods_names.append('Elliptic Envelope')
Outliers_methods_X_Y['X']['Elliptic Envelope'] = Ellipt_Envelop(X,Y)[0]
Outliers_methods_X_Y['Y']['Elliptic Envelope'] = Ellipt_Envelop(X,Y)[1]

##### Isolation Forest #####

def Iso_Forest(X, Y):
    Iso_For = IsolationForest(contamination = 0.2, max_samples=30)
    outlier_pred = Iso_For.fit_predict(Y)

    mask = outlier_pred != -1
    X, Y = X[mask, :], Y[mask]
    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)
    linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean())    
    return X, Y, abs(linear_reg_scores['test_score']).mean()



Outliers_methods_errors = np.append(Outliers_methods_errors,Iso_Forest(X,Y)[2])
Outliers_methods_names.append('Isolation Forest')
Outliers_methods_X_Y['X']['Isolation Forest'] = Iso_Forest(X,Y)[0]
Outliers_methods_X_Y['Y']['Isolation Forest'] = Iso_Forest(X,Y)[1]


###################################################### Best outlier removal method ######################################################

Best_outlier_rem_method = Outliers_methods_names[np.argmin(Outliers_methods_errors)]
print('Best Outlier Removal Method: ', Best_outlier_rem_method)

X = Outliers_methods_X_Y['X'][Best_outlier_rem_method]
Y = Outliers_methods_X_Y['Y'][Best_outlier_rem_method]

###################################################### Testing multiple predictors with k-fold cross validation after outlier removal ######################################################

# Data normalization
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)



k = 30 # Number of splits in cross validation



#### Ordinary Least Squares ####
Regressions_names.append('Ordinary Least Squares Regression')
lin_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score=True) 
lin_reg_MSE = abs(lin_reg_scores['test_score'].mean())

Predictors.append(linear_model.LinearRegression())
MSE_array = np.append(MSE_array,lin_reg_MSE)

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

Predictors.append(linear_model.Ridge(alpha = ridge_alpha))
MSE_array = np.append(MSE_array,ridge_reg_MSE)

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

Predictors.append(lasso)
MSE_array = np.append(MSE_array,lasso_reg_MSE)

#### Elastic-Net Regression ####
Regressions_names.append('Elastic-Net Regression')
# Generate array of alphas and l1-ratios
elastic_alphas = []
elastic_l1_ratios = []
alpha = 1
l1_ratio = 1
for i in range(200):
    alpha = alpha/1.05
    elastic_alphas.append(alpha)
    l1_ratio = l1_ratio/1.05
    elastic_l1_ratios.append(l1_ratio)


elastic_reg = linear_model.ElasticNetCV(alphas = elastic_alphas,l1_ratio=elastic_l1_ratios, random_state = 0, cv = k).fit(X, np.ravel(Y))
elastic_alpha = elastic_reg.alpha_ # Choose alpha that best fits the data
elastic_l1_ratio = elastic_reg.l1_ratio_ # Choose l1-ratio that best fits the data


elastic_net = linear_model.ElasticNet(alpha = elastic_alpha, l1_ratio = elastic_l1_ratio)
elastic_net_reg_scores = cross_validate(elastic_net, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
elastic_net_reg_MSE = abs(elastic_net_reg_scores['test_score'].mean())

Predictors.append(elastic_net)
MSE_array = np.append(MSE_array,elastic_net_reg_MSE)

#### Least Angle Regression ####
Regressions_names.append('Least Angle Regression')

LARS_reg_scores = cross_validate(linear_model.Lars(normalize = False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
LARS_reg_MSE = abs(LARS_reg_scores['test_score'].mean())

Predictors.append(linear_model.Lars(normalize = False))
MSE_array = np.append(MSE_array,LARS_reg_MSE)

#### Lasso Least Angle Regression ####
Regressions_names.append('Lasso Least Angle Regression')

lassoLars_reg = linear_model.LassoLarsCV(cv = k, normalize=False).fit(X, np.ravel(Y))
lassoLars_alpha = lassoLars_reg.alpha_ # Choose alpha that best fits the data

lassoLars_reg_scores = cross_validate(linear_model.LassoLars(alpha = lassoLars_alpha, normalize=False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
lassoLars_reg_MSE = abs(lassoLars_reg_scores['test_score'].mean())

Predictors.append(linear_model.LassoLars(alpha = lassoLars_alpha, normalize=False))
MSE_array = np.append(MSE_array,lassoLars_reg_MSE)

#### Orthogonal Matching Pursuit Regression ####
Regressions_names.append('Orthogonal Matching Pursuit Regression')

OMP_reg = linear_model.OrthogonalMatchingPursuitCV(cv = k, normalize=False, max_iter=10).fit(X, np.ravel(Y))
OMP_n_zero_coef = OMP_reg.n_nonzero_coefs_ # Choose number of non zero coefficients that best fits the data

OMP_reg_scores = cross_validate(linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = OMP_n_zero_coef, normalize=False), X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
OMP_reg_MSE = abs(OMP_reg_scores['test_score'].mean())

Predictors.append(linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = OMP_n_zero_coef, normalize=False))
MSE_array = np.append(MSE_array,OMP_reg_MSE)

#### Selection of best estimator to do prediction ####


min_MSE = np.argmin(MSE_array)

print(MSE_array)
print(min_MSE)
print('Best Predictor: ', Regressions_names[min_MSE])

predictor = Predictors[min_MSE]

# Making the prediction
train_predictor = predictor.fit(X, Y)
y_predict = train_predictor.predict(X_test)
np.save('Y_Predicted2.npy', y_predict)
print(np.shape(y_predict))