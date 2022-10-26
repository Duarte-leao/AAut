#from matplotlib import pyplot as plt
#import numpy as np
#X= np.load('Xtrain_Classification1.npy')
#Y= np.load('ytrain_Classification1.npy')


#from sklearn import svm
#X = [[0, 0], [1, 1]]
#y = [0, 1]
#clf = svm.SVC()
#clf.fit(X, y)

#clf.predict([[2., 2.]])




#print(X[600].reshape((30,30,3)))
#plt.imshow(X[233].reshape(30,30,3))
#plt.show()
#print(Y[233])

#from numpy.random import rand
#from matplotlib.pyplot import imshow

#img = rand(5, 5)  # Creating a 5x5 matrix of random numbers.

# Use bilinear interpolation (or it is displayed as bicubic by default).
#plt.imshow(img, interpolation="nearest")  
#plt.show()





"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 22/10/2022
Descrição: Classificação Parte 1
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
#import tensorflow_addons as tfa
#from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
#from imblearn.pipeline import Pipeline as imbpipeline
#from imblearn.keras import BalancedBatchGenerator
#from imblearn.keras import balanced_batch_generator
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import NearMiss


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)[1]
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)[1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)[1]
    precision = true_positives / (predicted_positives)
    recall = true_positives / (possible_positives)
    f1_val = 2*(precision*recall)/(precision+recall)

    
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # # print(true_positives)
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    # recall = true_positives / (possible_positives + K.epsilon())
    # f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

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

#Monotorização da função de custo
def Plots(model_train):
    f1_score = model_train.history['f1_score']
    val_f1_score = model_train.history['val_f1_score']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(len(f1_score))
    plt.plot(epochs, f1_score, 'g', label='Training f1 score')
    plt.plot(epochs, val_f1_score, 'r', label='Validation f1 score')
    plt.title('Training and validation f1 score')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

X = np.load('Xtrain_Classification1.npy')
Y = np.load('ytrain_Classification1.npy')
# X_test = np.load('Xtest_Regression1.npy')

# print(np.count_nonzero(Y==0))
# print(np.count_nonzero(Y==1))
# train_set = X.reshape(-1, 30, 30, 3)
# plt.imshow(train_set[500])
# plt.show()
#### SPLIT DATA   ####
# Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.20,random_state=2)

# Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,ytrain, test_size=0.10,random_state=2)

# Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.20, shuffle=True)

Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.20, random_state=2)
# Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,ytrain, test_size=0.10, shuffle=True)

####  NORMALIZE DATA   ####

Xtrain = Xtrain/255
Xval = Xval/255
# Xtest = Xtest/255

#Reshape
Xtrain = Xtrain.reshape(-1, 30, 30, 3)
Xval = Xval.reshape(-1, 30, 30, 3)
# Xtest = Xtest.reshape(-1, 30, 30, 3)

#print(Xtrain)



# #Imprimir resultados 
# print('Test loss:', test_eval[0])
# print('Test f1:', test_eval[1])
# Plots(cnn1_train)

#### Modelo 2 ####









# plt.figure()
# plt.imshow(Xtrain[500])
# plt.show()



# usar a função predict

#Imprimir resultados 
# print('Test loss:', test_eval[0])
# print('Test f1:', test_eval[1])
# Plots(cnn1_train)





##############Logistic Regression###################

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(Xtrain,ytrain)
#model.predict(Xval)
#model.score(Xval,yval)