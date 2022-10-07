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
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from statistics import mean
import matplotlib.pyplot as plt

# Load the data set

X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')


################################## IQR #####################################
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
    
# print("Index list x \n", index_list_x)
# print("Index list y \n", index_list_y)
    
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

# print("X_cleaned:",df_x_cleaned.shape)
# print("y_cleaned:",df_y_cleaned.shape)

X_IQR = df_x_cleaned.to_numpy()
Y_IQR = df_y_cleaned.to_numpy()

####################################### Outlier removal via 100-fold cross validation ###########################################
k=100
flag=1

while flag != 0:
    linear_reg = linear_model.LinearRegression()
    linear_reg_scores = cross_validate(linear_reg, X, Y, cv = k, scoring = 'neg_mean_squared_error', return_train_score = True)
    linear_reg_MSE = abs(linear_reg_scores['test_score'])
    print(len(linear_reg_MSE))
    print(len(linear_reg_MSE))
    if linear_reg_MSE.max() > 1:
        print('Outlier:', np.argmax(linear_reg_MSE), 'MSE:',linear_reg_MSE.max())
        X=np.delete(X,np.argmax(linear_reg_MSE),0)
        Y=np.delete(Y,np.argmax(linear_reg_MSE),0)
    else:
        flag = 0

    k -=1

print("X_cleaned:",X.shape)
print("y_cleaned:",Y.shape)