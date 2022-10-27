
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

X = np.load('Xtrain_Classification1.npy')
Y = np.load('ytrain_Classification1.npy')

Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.20, random_state=2)

Xtrain = Xtrain/255
Xval = Xval/255

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

gnb = GaussianNB()
Xtrain_not_CNN, ytrain_not_CNN = data_balance_generator(Xtrain, train_labels, False)
print(np.shape(Xtrain))
print(np.shape(Xtrain_not_CNN))
pred_y = gnb.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)
pred_y2 = gnb.fit(Xtrain,ytrain).predict(Xval)

print(met.confusion_matrix(yval,pred_y))
print(met.f1_score(yval,pred_y))
print(met.f1_score(yval,pred_y2))

