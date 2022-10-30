"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 30/10/2022
Descrição: Classificação Parte 2
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score as BACC
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X = np.load('Xtrain_Classification2.npy')
Y = np.load('Ytrain_Classification2.npy')
X_test = np.load('Xtest_Classification2.npy')

print(np.shape(X))
print(np.shape(Y))
print(np.shape(X_test))

print(X)
print(Y)
print(X_test)

#Normalize data   

X = X/255
X_test = X_test/255