from sklearn.naive_bayes import GaussianNB
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score as BACC
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.keras import BalancedBatchGenerator
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X = np.load('Xtrain_Classification1.npy')
Y = np.load('ytrain_Classification1.npy')

Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15)

# Xtrain = Xtrain/255
# Xval = Xval/255

train_labels = keras.utils.to_categorical(ytrain, num_classes=2)

def traduz_vec(prob_vec):
    # print(prob_vec)

    pred_labels=[]
    # for i in range(len(prob_vec)):
    #     if prob_vec[i,0] > 0.5:
    #         pred_labels.append(0) 
    #     elif prob_vec[i,1] >= 0.5:
    #         pred_labels.append(1) 
    for i in range(len(prob_vec)):
        if prob_vec[i,0] > prob_vec[i,1]:
            pred_labels.append(0) 
        elif prob_vec[i,0] < prob_vec[i,1]:
            pred_labels.append(1) 
            
    pred_labels = np.ravel(pred_labels)
    return pred_labels

def data_balance_generator(Xtrain, train_labels, CNN = True):
    # plt.figure()
    # plt.imshow(Xtrain[500])
    if CNN:
        Xtrain = Xtrain.reshape(-1, 2700)

    training_generator = balanced_batch_generator(
        Xtrain, train_labels, sampler=RandomOverSampler(sampling_strategy=1), batch_size=1)


    Xtrain = []
    train_labels = []
    for i, el in enumerate(training_generator[0]): 
        Xtrain.append(el[0])
        train_labels.append(el[1])
        if i == training_generator[1]:
            break
    Xtrain = np.vstack(Xtrain)
    train_labels = np.vstack(train_labels)

    if CNN: 
        Xtrain = Xtrain.reshape(-1, 30, 30, 3)
    else:
        train_labels = traduz_vec(train_labels)

    # plt.figure()
    # plt.imshow(Xtrain[500])
    # plt.show()
    return Xtrain, train_labels

def class_weights(y_train):
    cls_wt = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)

    class_weights = {0: cls_wt[0], 1:cls_wt[1]}
    return class_weights
# gnb = GaussianNB()
Xtrain_not_CNN, ytrain_not_CNN = data_balance_generator(Xtrain, train_labels, False)
# f1 = 0
# k = 0
# for i in range(2,100):
#     gnb = KNeighborsClassifier(n_neighbors=i, weights= 'distance')


#     # print(np.shape(Xtrain))
#     # print(np.shape(Xtrain_not_CNN))
#     pred_y = gnb.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)
#     # pred_y2 = gnb.fit(Xtrain,ytrain).predict(Xval)

#     # print(met.confusion_matrix(yval,pred_y))
#     print(i)
#     if f1<met.f1_score(yval,pred_y):
#         f1 = met.f1_score(yval,pred_y)
#         k = i
# # print(met.f1_score(yval,pred_y2))

# print(k)
# gnb = KNeighborsClassifier(n_neighbors=51, weights= 'distance')

# # 92
# # 51
# pred_y = gnb.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)
# print(met.f1_score(yval,pred_y))

# gnb = DecisionTreeClassifier()


# pred_y = gnb.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)

# gnb2 = DecisionTreeClassifier(class_weight = class_weights(ytrain))

# pred_y2 = gnb2.fit(Xtrain,ytrain).predict(Xval)

# print(met.f1_score(yval,pred_y))
# print(met.f1_score(yval,pred_y2))






##############Logistic Regression###################
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# param_grid = {'C': [0.04, 0.05], 'max_iter': [2700]}
# grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose=3)
# grid.fit(Xtrain,ytrain)
# print(grid.best_params_)
# print(grid.best_estimator_)
# grid_predictions= grid.predict(Xval)
# print('Logistic Regression F1 Score')
# print(f1_score(yval,grid_predictions))

#Melhor Resultado para os parametros de LogisticRegression
#{'C': 0.05, 'max_iter': 2700}
# LogReg = LogisticRegressionCV(Cs=[0.01, 0.05, 0.1] ,class_weight=class_weights(ytrain))
# # pred_y = a.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)

# pred_y = LogReg.fit(Xtrain,ytrain).predict(Xval)
# print(LogReg.C_)
# pred_y = LogisticRegression(C = 0.01, class_weight=class_weights(ytrain)).fit(Xtrain,ytrain).predict(Xval)
# # print(pred_y)
#Logistic Regression F1 Score
#0.5927272727272729



##############Support vector machine###################

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score 
# param_grid = { 'gamma':[0.008,0.009],'C': [0.5,  3], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'degree': [3]}
# param_grid = { 'gamma':['scale'],'C': [4,5], 'kernel': [ 'rbf']}

# param_grid = {'C': [0.5,0.6]}

# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3,cv=3,scoring='f1')
# grid = GridSearchCV(LinearSVC(class_weight=class_weights(ytrain)), param_grid, refit = False, verbose=3,cv=3)

# grid.fit(Xtrain,ytrain)
# print(grid.best_params_)
# print(grid.best_estimator_)
# grid_predictions= grid.predict(Xval)
# print('Support vector machine F1 Score')
# print(f1_score(yval,grid_predictions))


svc = SVC(class_weight=class_weights(ytrain),C=3, gamma='scale', kernel='rbf')
svc2 = SVC(class_weight=class_weights(ytrain), C=4, gamma='scale', kernel='rbf')
pred_y = svc.fit(Xtrain,ytrain).predict(Xval)
pred_y2 = svc2.fit(Xtrain,ytrain).predict(Xval)

# {'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}
print(met.f1_score(yval,pred_y))
print(met.f1_score(yval,pred_y2))