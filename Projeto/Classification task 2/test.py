"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 30/10/2022
Descrição: Classificação Parte 2
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
# from sklearn.metrics import balanced_accuracy_score as BACC
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from scikeras.wrappers import KerasClassifier
import math
import warnings

X = np.load('Xtrain_Classification2.npy')
Y = np.load('Ytrain_Classification2.npy')
X_test = np.load('Xtest_Classification2.npy')


#Normalize data   

X = X/255
X_test = X_test/255

# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15)
Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)

# print(np.sum(np.shape(Xtrain)))

# print(np.count_nonzero(Y==0))
# print(np.count_nonzero(Y==1))
# print(np.count_nonzero(Y==2))

def undo_one_hot_encoding(vec):   
    arr = np.zeros(len(vec))
    for i in range(len(vec)):
        arr[i] = np.argmax(vec[i])
    return arr

def Plots(model_train):
    # BACC = model_train.history['BACC']
    # val_BACC = model_train.history['val_BACC']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(len(loss))
    # plt.plot(epochs, BACC, 'g', label='Training f1 score')
    # plt.plot(epochs, val_BACC, 'r', label='Validation f1 score')
    # plt.title('Training and validation f1 score')
    # plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []
    
    nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment
    
    return layers

def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, metrics = None):
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers):
        if i==1:
            model.add(Dense(first_layer_nodes, input_dim=X.shape[1], activation=activation_func))
        else:
            model.add(Dense(n_nodes[i-1], activation=activation_func))
            

    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = metrics)
    
    return model

def sampling_method(method, random_state = None):
    if method == 0:
        balancing_method = SMOTE(random_state= random_state)
    elif method == 1:
        balancing_method = RandomOverSampler(random_state= random_state)

    return balancing_method

def balance_data(x, y, method):

    sampling = sampling_method(method,1)
    X, Y = sampling.fit_resample(x, y)

    return X, Y

def BACC(y_true, y_pred, *, sample_weight=None, adjusted=False):

    # print(y_true)
    # print(y_pred)
    shape = np.shape(y_pred)
    if np.sum(shape)>shape[0]:
        y_true = undo_one_hot_encoding(y_true)
        y_pred = undo_one_hot_encoding(y_pred)

    C = met.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn("y_pred contains classes not in y_true")
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    print(score)
    return score

def hyperparameters_tunningNN(x_training, y_training, balance):
     
    scoring = {'score': met.make_scorer(BACC)}
    activation_funcs = ['relu'] 
    # activation_funcs = ['sigmoid', 'relu', 'tanh'] 

    # y_training = tf.keras.utils.to_categorical(y_training, num_classes=3)
    # print(y_training)
    if balance == 1:
        model =  KerasClassifier(build_fn=createmodel, verbose = False)
        # model =  KerasClassifier(build_fn=createmodel, activation_func='sigmoid',first_layer_nodes=32, last_layer_nodes=4,n_layers=2,  verbose = False)
        param_grid = dict(classifier__n_layers=[4], classifier__first_layer_nodes = [128,256], classifier__last_layer_nodes = [32,16],  classifier__activation_func = activation_funcs,  classifier__batch_size = [300], classifier__epochs = [40,50])
        # param_grid = dict(classifier__n_layers=[2,3,4], classifier__first_layer_nodes = [64,32,16], classifier__last_layer_nodes = [8, 4],  classifier__activation_func = activation_funcs,  classifier__batch_size = [100,200], classifier__epochs = [20,30,60])
        pipeline = imbpipeline(steps = [['sampling', sampling_method(0,1)],['classifier', model]])
        gs_cv = GridSearchCV(pipeline , param_grid, scoring=scoring, cv= 4, refit="score", verbose=2)
        gs_cv.fit(x_training,y_training)
        print(gs_cv.best_params_)
        print(gs_cv.best_score_)


def hyperparameters_tunning(x_training, y_training, balance):
    
    scoring = {'score': met.make_scorer(BACC)}

    # if balance == 1:
    #     parameters={ 'classifier__n_neighbors':[5,10,20,35,50],  'classifier__weights':[  'uniform', 'distance'], 'classifier__algorithm':[ 'auto','ball_tree','kd_tree']}
    #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', KNeighborsClassifier())])
    #     gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
    #     gs_cv.fit(x_training,y_training)
    #     print(gs_cv.best_params_)
    #     print(gs_cv.best_score_)
    if balance == 1:
        parameters={ 'classifier__C':[5,10,20,35,50],  'classifier__kernel':[  'uniform', 'distance'], 'classifier__algorithm':[ 'auto','ball_tree','kd_tree']}
        pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', KNeighborsClassifier())])
        gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
        gs_cv.fit(x_training,y_training)
        print(gs_cv.best_params_)
        print(gs_cv.best_score_)

    
# hyperparameters_tunningNN(Xtrain,ytrain,1)
# hyperparameters_tunning(Xtrain,ytrain,1)


Xtrain, ytrain = balance_data(Xtrain, ytrain, 0)

train_labels = tf.keras.utils.to_categorical(ytrain, num_classes=3)
valid_labels = tf.keras.utils.to_categorical(yval, num_classes=3)

# model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100,),activation = 'relu', learning_rate_init=0.001)
my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
model = createmodel(n_layers=2, first_layer_nodes=100, last_layer_nodes=100, activation_func='relu')
# model = createmodel(n_layers=4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu')
NN_train = model.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels), epochs= 50,batch_size=200, callbacks= my_callbacks)
# Plots(NN_train)

# model.fit(Xtrain, ytrain)

y_pred = model.predict(Xval)
# print(y_pred)
print(BACC(valid_labels, y_pred)) #tf NN
# print(BACC(yval, y_pred))


# {'classifier__alpha': 0.001, 'classifier__hidden_layer_sizes': (14,), 'classifier__learning_rate_init': 0.001}

