"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 30/10/2022
Descrição: Classificação Parte 2
*******************************************************************************************
"""

import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import math
import warnings

##################################################   Functions   ##################################################
# Preprocessing methods
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


# Score
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

# Generate MLP model
def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []
    
    nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment
    
    return layers

def MLP(n_layers, first_layer_nodes, last_layer_nodes, activation_func, metrics = None):
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

def undo_one_hot_encoding(vec):   
    arr = np.zeros(len(vec))
    for i in range(len(vec)):
        arr[i] = np.argmax(vec[i])
    return arr

# Plot for Neural Network
def Plot(model_train):
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def hyperparameter_tunning(x_training, y_training, Classifiers):
    scoring = {'score': met.make_scorer(BACC)}
    for i, classifier in enumerate(Classifiers):
        # Define parameters to do grid search
        if i == 0: #MLP
            param_grid = dict(classifier__n_layers=[2,3,4], classifier__first_layer_nodes = [128,64,32,16], classifier__last_layer_nodes = [32,16,8,4],  classifier__activation_func = ['sigmoid', 'relu', 'tanh'],  classifier__batch_size = [100,200,300], classifier__epochs = [30,40,50])
        elif i == 1: #KNN
            param_grid={ 'classifier__n_neighbors':[2,4,5,10,20,35,50],  'classifier__weights':[  'uniform', 'distance'], 'classifier__algorithm':[ 'auto','ball_tree','kd_tree']}
        elif i == 2: #SVC
            param_grid={ 'classifier__C':[0.1, 0.5,1, 2, 4,7],  'classifier__kernel':['poly', 'rbf', 'sigmoid'], 'classifier__gamma':[ 'auto', 'scale', 0.1,0.03,0.07,0.5]}
        elif i == 3: #DT
            param_grid={ 'classifier__criterion':['gini', 'entropy', 'log_loss'],  'classifier__splitter':[ 'best', 'random'], 'classifier__min_samples_split':[ 2,5,10]}
        elif i == 4: #RF
            param_grid={  'classifier__n_estimators':[50, 100, 150, 200], 'classifier__criterion':['gini', 'entropy', 'log_loss'], 'classifier__min_samples_split':[3, 2]}
        elif i == 5: #Logistic Reg
            param_grid={  'classifier__penalty':['l1', 'l2', 'elasticnet', 'none'], 'classifier__dual':[False, True], 'classifier__C':[1, 2 ,5], 'classifier__l1_ratio':[0.5,0.7], 'classifier__solver':['lbfgs','liblinear','saga']}
        
        pipeline = imbpipeline(steps = [('sampling', sampling_method(1,1)),('classifier', classifier)])
        # Do grid search
        if i == 0:
            gs_cv = GridSearchCV(pipeline , param_grid, scoring=scoring, cv= 4,refit="score", verbose=2)
        else:
            gs_cv = GridSearchCV(pipeline , param_grid, scoring=scoring, cv= 4,refit="score", n_jobs=-1, verbose=2)

        gs_cv.fit(x_training,y_training)
        print(gs_cv.best_params_)
        print(gs_cv.best_score_)

        # Best parameters
        # (n_layers= 4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu', epochs= 40, batch_size=100) #MLP
        # (n_neighbors=4, algorithm='auto', weights= 'distance') #KNN
        # (C = 7, kernel = 'rbf' ,gamma = .07) #SVC
        # (criterion= 'gini', min_samples_split= 5, splitter= 'best') #DT
        # (criterion= 'entropy', min_samples_split= 2, n_estimators= 150) #RF
        # (penalty = 'l2', C = 2, solver = 'lbfgs',max_iter=1000, n_jobs=-1) #Logistic reg

def best_model(Classifiers, params):
    Scores = []
    scoring = {'score': met.make_scorer(BACC)}
    classifier = Classifiers[1]
    # for i, classifier in enumerate(Classifiers):
    # pipeline = imbpipeline(steps = [('classifier', classifier)])
    pipeline = imbpipeline(steps = [('sampling', sampling_method(1,1)),('classifier', classifier)])
        # if i == 0:
        #     classifier_cv = GridSearchCV(pipeline , params[i], scoring=scoring, cv= 4,refit="score", verbose=2)
        # else:
    classifier_cv = GridSearchCV(pipeline , params[1], scoring=scoring, cv= 5,refit="score", n_jobs=-1, verbose=2)
        # classifier_cv.fit(Xtrain, ytrain)
    classifier_cv.fit(X, Y)
    Scores.append(classifier_cv.best_score_)
    print(Scores)
    # Scores
    # [0.8820236447035272, 0.9096238273878492, 0.8578112489289657, 0.7784271347421321, 0.8736156059194251, 0.7613127666499254]
    return np.argmax(Scores)

def prediction(Classifiers_best,Classifiers, params):
    chosen_model = best_model(Classifiers, params) 
    # print(chosen_model)
    chosen_model = 4 # Because RF yields the best results
    classifier = Classifiers_best[chosen_model]
    X_train, y_train = balance_data(Xtrain, ytrain, 1)
    # X_, Y_ = balance_data(X, Y, 1)
    X_, Y_ = balance_data(Xtrain, ytrain, 1)
    # X_, Y_ = Xtrain, ytrain
    if chosen_model == 0:
        train_labels = tf.keras.utils.to_categorical(y_train, num_classes=3)
        valid_labels = tf.keras.utils.to_categorical(yval, num_classes=3)
        my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
        MLP_train = classifier.fit(X_train, train_labels,validation_data=(Xval, valid_labels), epochs= 40,batch_size=100, callbacks= my_callbacks)
        # Plot(MLP_train)    
        y_pred = classifier.predict(X_test)
        y_pred = undo_one_hot_encoding(y_pred)

        y_val_pred = classifier.predict(Xval)
        y_val_pred = undo_one_hot_encoding(y_val_pred)
    else:
        classifier.fit(X_, Y_)
        y_pred = classifier.predict(X_test)
        y_val_pred = classifier.predict(Xval)

    BACC(yval, y_val_pred)
    return y_pred

##################################################   Main   ##################################################
X = np.load('Xtrain_Classification2.npy')
Y = np.load('Ytrain_Classification2.npy')
X_test = np.load('Xtest_Classification2.npy')


#Normalize data   

X = X/255
X_test = X_test/255

# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.2)
Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)


    
Classifiers = [KerasClassifier(build_fn=MLP, verbose = False), KNeighborsClassifier(n_jobs=-1), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1),  LogisticRegression(n_jobs=-1)]

best_params = [dict(classifier__n_layers=[ 4], classifier__first_layer_nodes=[128], classifier__last_layer_nodes=[16], classifier__activation_func=['relu'], classifier__epochs=[ 40], classifier__batch_size=[100]),
                dict(classifier__n_neighbors=[4], classifier__algorithm=['auto'], classifier__weights=[ 'distance']), dict(classifier__C =[ 7], classifier__kernel =[ 'rbf'], classifier__gamma =[ .07]),
                dict(classifier__criterion=[ 'gini'], classifier__min_samples_split=[ 5], classifier__splitter=[ 'best']), dict(classifier__criterion=[ 'entropy'], classifier__min_samples_split=[ 2], classifier__n_estimators=[ 150]),
                dict(classifier__penalty =[ 'l2'], classifier__C =[ 2], classifier__solver =[ 'lbfgs'], classifier__max_iter=[1000], classifier__n_jobs=[-1])]

Classifiers_with_best_params = [MLP(n_layers= 4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu'),
                                KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights= 'distance', n_jobs=-1), SVC(C = 7, kernel = 'rbf' ,gamma = .07),
                                DecisionTreeClassifier(criterion= 'gini', min_samples_split= 5, splitter= 'best'), RandomForestClassifier(criterion= 'entropy', min_samples_split= 2, n_estimators= 150, n_jobs=-1),
                                LogisticRegression(penalty = 'l2', C = 2, solver = 'lbfgs',max_iter=1000, n_jobs=-1)]

# y_sum_predictions = 0
# for i in range(50):
#     print(i)
# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.2)

# y_predict_aux = prediction(Classifiers_with_best_params, Classifiers, best_params)
# y_sum_predictions += tf.keras.utils.to_categorical(y_predict_aux, num_classes=3)

# y_predict = undo_one_hot_encoding(y_sum_predictions)
y_predict = prediction(Classifiers_with_best_params, Classifiers, best_params)

# np.save('Y_Predicted.npy', y_predict)
# np.save('Y_Predicted4.npy', y_predict)

# y_pred1 = np.load('Y_PredictedRF.npy')
# y_pred2 = np.load('Y_Predicted_mlp.npy')

# BACC(y_pred2, y_pred1)
# print(np.shape(X_test))
# print(np.shape(y_predict))