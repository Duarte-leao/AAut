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

###################################################### Outlier removal ######################################################
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

    return X, Y
# X, Y = outlier_removal_IQR(X, Y)
# linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
# print(abs(linear_reg_scores['test_score']).mean())

##### Outlier removal via 100-fold cross validation #####
def outlier_removal_CV(X, Y):
    while True:
        linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = X.shape[0], scoring = 'neg_mean_squared_error', return_train_score = True)
        linear_reg_MSE = abs(linear_reg_scores['test_score'])
        if linear_reg_MSE.max() > 1:
            # print('Outlier MSE:',linear_reg_MSE.max())
            X=np.delete(X,np.argmax(linear_reg_MSE),0)
            Y=np.delete(Y,np.argmax(linear_reg_MSE),0)
        else:
            break


    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)

    return X, Y

# X, Y = outlier_removal_CV(X, Y)
# linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
# print(abs(linear_reg_scores['test_score']).mean())

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

    return X, Y
    
# X, Y = LOF(X, Y)
# linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
# print(abs(linear_reg_scores['test_score']).mean())

##### Elliptic Envelope #####

def Ellipt_Envelop(X, Y):
    Ellip_Env = EllipticEnvelope(contamination = 0.2)
    outlier_pred = Ellip_Env.fit_predict(Y)

    mask = outlier_pred != -1
    X, Y = X[mask, :], Y[mask]
    # print("X_cleaned:",X.shape)
    # print("y_cleaned:",Y.shape)
    # linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean()) 
    return X, Y

# X, Y = Ellipt_Envelop(X, Y)
# linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
# print(abs(linear_reg_scores['test_score']).mean())


##### Isolation Forest #####

def Iso_Forest(X, Y):
    Iso_For = IsolationForest(contamination = 0.2, max_samples=30)
    outlier_pred = Iso_For.fit_predict(Y)

    mask = outlier_pred != -1
    X, Y = X[mask, :], Y[mask]
    print("X_cleaned:",X.shape)
    print("y_cleaned:",Y.shape)
    # linear_reg_scores = cross_validate(linear_model.LinearRegression(), X, Y, cv =5 , scoring = 'neg_mean_squared_error', return_train_score = True)
    # print(abs(linear_reg_scores['test_score']).mean())    
    return X, Y

# X, Y = Iso_Forest(X,Y)

###################################################### Robust regressors to outliers ######################################################
def Robust_regressors(X,Y):
    # Data normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    

    # Testing multiple predictors with k-fold cross validation

    k = 10 # Number of splits in cross validation

    ##### KNN Regressor #####

    KNN_reg = neighbors.KNeighborsRegressor(n_neighbors=35,weights='distance',algorithm='brute')
    KNN_reg_scores = cross_validate(KNN_reg, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    KNN_reg_MSE = abs(KNN_reg_scores['test_score'])
    print('KNN MSE:', KNN_reg_MSE.mean())

    ##### Huber Regressor #####

    Huber_reg = linear_model.HuberRegressor(max_iter=500, epsilon=20)
    Huber_reg_scores = cross_validate(Huber_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    Huber_reg_MSE = abs(Huber_reg_scores['test_score'].mean())
    print('Huber Regressor MSE:', Huber_reg_MSE)

    ##### RANSACRegressor #####

    RANSAC_reg = linear_model.RANSACRegressor(min_samples=20,residual_threshold=1,loss='squared_error')
    RANSAC_reg_scores = cross_validate(RANSAC_reg, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    RANSAC_reg_MSE = abs(RANSAC_reg_scores['test_score'].mean())
    print('RANSAC Regressor MSE:', RANSAC_reg_MSE)

    ##### TheilSen Regressor #####

    TheilSen_reg = linear_model.TheilSenRegressor(n_subsamples=20)
    TheilSen_reg_scores = cross_validate(TheilSen_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    TheilSen_reg_MSE = abs(TheilSen_reg_scores['test_score'].mean())
    print('TheilSen Regressor MSE:', TheilSen_reg_MSE)

    # ##### Random Forest Regressor #####

    RdmFor_reg = ensemble.RandomForestRegressor()
    RdmFor_reg_scores = cross_validate(RdmFor_reg, X, np.ravel(Y), cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    RdmFor_reg_MSE = abs(RdmFor_reg_scores['test_score'].mean())
    print('Random Forest MSE:', RdmFor_reg_MSE)