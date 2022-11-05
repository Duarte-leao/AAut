# """
# *******************************************************************************************
# Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
# IST     Data: 30/10/2022
# Descrição: Classificação Parte 2
# *******************************************************************************************
# """

# import keras.backend as K
# import tensorflow as tf
# # from sklearn.metrics import balanced_accuracy_score as BACC
# from tensorflow import keras
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split
# import sklearn.metrics as met
# from imblearn.pipeline import Pipeline as imbpipeline
# from imblearn.keras import balanced_batch_generator
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE
# from sklearn.utils import class_weight
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# import math
# import warnings

# X = np.load('Xtrain_Classification2.npy')
# Y = np.load('Ytrain_Classification2.npy')
# X_test = np.load('Xtest_Classification2.npy')


# #Normalize data   

# X = X/255
# X_test = X_test/255

# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.2, random_state=2)
# # Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)

# # print(np.shape(Xval))
# # print(np.shape(X_test))

# # print(np.count_nonzero(Y==0))
# # print(np.count_nonzero(Y==1))
# # print(np.count_nonzero(Y==2))

# def undo_one_hot_encoding(vec):   
#     arr = np.zeros(len(vec))
#     for i in range(len(vec)):
#         arr[i] = np.argmax(vec[i])
#     return arr

# def Plot(model_train):
#     # BACC = model_train.history['BACC']
#     # val_BACC = model_train.history['val_BACC']
#     loss = model_train.history['loss']
#     val_loss = model_train.history['val_loss']
#     epochs = range(len(loss))
#     # plt.plot(epochs, BACC, 'g', label='Training f1 score')
#     # plt.plot(epochs, val_BACC, 'r', label='Validation f1 score')
#     # plt.title('Training and validation f1 score')
#     # plt.legend()
#     plt.figure()
#     plt.plot(epochs, loss, 'g', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#     plt.show()

# def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
#     layers = []
    
#     nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
#     nodes = first_layer_nodes
#     for i in range(1, n_layers+1):
#         layers.append(math.ceil(nodes))
#         nodes = nodes + nodes_increment
    
#     return layers

# def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, metrics = None):
#     model = Sequential()
#     n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
#     for i in range(1, n_layers):
#         if i==1:
#             model.add(Dense(first_layer_nodes, input_dim=X.shape[1], activation=activation_func))
#         else:
#             model.add(Dense(n_nodes[i-1], activation=activation_func))
            

#     model.add(Dense(3, activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = metrics)
    
#     return model

# def sampling_method(method, random_state = None):
#     if method == 0:
#         balancing_method = SMOTE(random_state= random_state)
#     elif method == 1:
#         balancing_method = RandomOverSampler(random_state= random_state)

#     return balancing_method

# def balance_data(x, y, method):
#     sampling = sampling_method(method,1)
#     X, Y = sampling.fit_resample(x, y)
#     return X, Y

# def BACC(y_true, y_pred, *, sample_weight=None, adjusted=False):

#     # print(y_true)
#     # print(y_pred)
#     shape = np.shape(y_pred)
#     if np.sum(shape)>shape[0]:
#         y_true = undo_one_hot_encoding(y_true)
#         y_pred = undo_one_hot_encoding(y_pred)

#     C = met.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         per_class = np.diag(C) / C.sum(axis=1)
#     if np.any(np.isnan(per_class)):
#         warnings.warn("y_pred contains classes not in y_true")
#         per_class = per_class[~np.isnan(per_class)]
#     score = np.mean(per_class)
#     if adjusted:
#         n_classes = len(per_class)
#         chance = 1 / n_classes
#         score -= chance
#         score /= 1 - chance
#     print(score)
#     return score

# def hyperparameters_tunningNN(x_training, y_training, balance):
     
#     scoring = {'score': met.make_scorer(BACC)}
#     activation_funcs = ['relu'] 
#     # activation_funcs = ['sigmoid', 'relu', 'tanh'] 

#     # y_training = tf.keras.utils.to_categorical(y_training, num_classes=3)
#     # print(y_training)
#     if balance == 1:
#         model =  KerasClassifier(build_fn=createmodel, verbose = False)
#         # model =  KerasClassifier(build_fn=createmodel, activation_func='sigmoid',first_layer_nodes=32, last_layer_nodes=4,n_layers=2,  verbose = False)
#         param_grid = dict(classifier__n_layers=[4], classifier__first_layer_nodes = [128,256], classifier__last_layer_nodes = [32,16],  classifier__activation_func = ['sigmoid', 'relu', 'tanh'] ,  classifier__batch_size = [200, 300], classifier__epochs = [30,40,50])
#         # param_grid = dict(classifier__n_layers=[2,3,4], classifier__first_layer_nodes = [128,64,32,16], classifier__last_layer_nodes = [32,16,8,4],  classifier__activation_func = ['sigmoid', 'relu', 'tanh'],  classifier__batch_size = [100,200,300], classifier__epochs = [30,40,50])
#         pipeline = imbpipeline(steps = [['sampling', sampling_method(0,1)],['classifier', model]])
#         gs_cv = GridSearchCV(pipeline , param_grid, scoring=scoring, cv= 4, refit="score", verbose=2)
#         gs_cv.fit(x_training,y_training)
#         print(gs_cv.best_params_)
#         print(gs_cv.best_score_)


# def hyperparameters_tunning(x_training, y_training, balance):
    
#     scoring = {'score': met.make_scorer(BACC)}

#     if balance == 1:
#         parameters={ 'classifier__n_neighbors':[4],  'classifier__weights':['distance'], 'classifier__algorithm':[ 'auto']}
#         # parameters={ 'classifier__n_neighbors':[5,10,20,35,50],  'classifier__weights':[  'uniform', 'distance'], 'classifier__algorithm':[ 'auto','ball_tree','kd_tree']}
#         pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', KNeighborsClassifier(n_jobs=-1))])
#         gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
#         # gs_cv.fit(X,Y)
#         gs_cv.fit(x_training,y_training)
#         print(gs_cv.best_params_)
#         print(gs_cv.best_score_)
#     # if balance == 1:
#     #     # parameters={ 'classifier__C':[0.1, 0.5,1, 2, 4,7],  'classifier__kernel':['poly', 'rbf', 'sigmoid'], 'classifier__gamma':[ 'auto', 'scale', 0.1,0.03,0.07,0.5]}
#     #     parameters={ 'classifier__C':[7],  'classifier__kernel':[ 'rbf'], 'classifier__gamma':[ 0.03,0.07,0.05]}
#     #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', SVC())])
#     #     gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
#     #     gs_cv.fit(x_training,y_training)
#     #     print(gs_cv.best_params_)
#     #     print(gs_cv.best_score_)
#     # if balance == 1:
#     #     parameters={ 'classifier__criterion':['gini', 'entropy', 'log_loss'],  'classifier__splitter':[ 'best', 'random'], 'classifier__min_samples_split':[ 2,5,10]}
#     #     # parameters={ 'classifier__criterion':['log_loss'],  'classifier__splitter':['random'], 'classifier__min_samples_split':[ 2,5,10],}
#     #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', DecisionTreeClassifier())])
#     #     gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
#     #     gs_cv.fit(x_training,y_training)
#     #     print(gs_cv.best_params_)
#     #     print(gs_cv.best_score_)
#     # if balance == 1:
#     #     # parameters={  'classifier__n_estimators':[50, 100, 150, 200], 'classifier__criterion':['gini', 'entropy', 'log_loss'], 'classifier__min_samples_split':[4, 2]}
#     #     parameters={  'classifier__n_estimators':[175, 150], 'classifier__criterion':['entropy'], 'classifier__min_samples_split':[4,2]}
#     #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', RandomForestClassifier(n_jobs=-1))])
#     #     gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
#     #     gs_cv.fit(x_training,y_training)
#     #     print(gs_cv.best_params_)
#     #     print(gs_cv.best_score_)
#     # if balance == 1:
#     #     # parameters={  'classifier__n_estimators':[50, 100, 150], 'classifier__criterion':['gini', 'entropy', 'log_loss'], 'classifier__min_samples_split':[ 2]}
#     #     # parameters={  'classifier__penalty':['l1', 'l2', 'elasticnet', 'none'], 'classifier__dual':[False, True], 'classifier__C':[1,5], 'classifier__l1_ratio':[0.5,0.7], 'classifier__solver':['liblinear','saga']}
#     #     parameters={  'classifier__penalty':['l1'], 'classifier__dual':[False], 'classifier__C':[1], 'classifier__solver':['saga']}
#     #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', LogisticRegression(n_jobs=-1))])
#     #     gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 4,refit="score",n_jobs=-1, verbose=2)
#     #     gs_cv.fit(x_training,y_training)
#     #     print(gs_cv.best_params_)
#     #     print(gs_cv.best_score_)
    

# # hyperparameters_tunningNN(Xtrain,ytrain,1)
# # hyperparameters_tunning(Xtrain,ytrain,1)


# Xtrain, ytrain = balance_data(Xtrain, ytrain, 0)

# train_labels = tf.keras.utils.to_categorical(ytrain, num_classes=3)
# valid_labels = tf.keras.utils.to_categorical(yval, num_classes=3)


# print(valid_labels+0)

# # model = KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights= 'distance', n_jobs=-1)
# # model.fit(Xtrain, ytrain)

# # y_pred = model.predict(Xval)
# # print(BACC(yval, y_pred))
# # model = SVC(C = 7, kernel = 'rbf' ,gamma = .07)
# # model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100,),activation = 'relu', learning_rate_init=0.001)
# # my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
# # model = createmodel(n_layers= 4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu')
# # model = createmodel(n_layers=6, first_layer_nodes=512, last_layer_nodes=32, activation_func='relu')
# # model = LogisticRegression(penalty = 'l1', C = 2, solver = 'saga',max_iter=1000, n_jobs=-1)
# # model = LogisticRegression(penalty = 'l2', C = 5, solver = 'lbfgs',max_iter=1000, n_jobs=-1)
# # NN_train = model.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels), epochs= 40,batch_size=100, callbacks= my_callbacks)
# # Plot(NN_train)

# # model.fit(Xtrain, ytrain)

# # Xval1, Xval2, yval1, yval2 = train_test_split(Xval, yval, test_size=.5, random_state=2)
# # valid_labels1 = tf.keras.utils.to_categorical(yval1, num_classes=3)
# # valid_labels2 = tf.keras.utils.to_categorical(yval2, num_classes=3)

# # y_pred1 = model.predict(Xval1)
# # y_pred2 = model.predict(Xval2)
# # y_pred = model.predict(Xval)
# # # print(y_pred)
# # print(BACC(valid_labels1, y_pred1)) #tf NN
# # print(BACC(valid_labels2, y_pred2)) #tf NN
# # print(BACC(valid_labels, y_pred)) #tf NN
# # print(BACC(yval, y_pred))

# # {'classifier__criterion': 'entropy', 'classifier__min_samples_split': 2, 'classifier__n_estimators': 150} RF
# # (n_layers=4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu')

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
        
        pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', classifier)])
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
    # Scores = []
    # scoring = {'score': met.make_scorer(BACC)}
    # for i, classifier in enumerate(Classifiers):
    #     pipeline = imbpipeline(steps = [('sampling', sampling_method(0,1)),('classifier', classifier)])
    #     if i == 0:
    #         classifier_cv = GridSearchCV(pipeline , params[i], scoring=scoring, cv= 4,refit="score", verbose=2)
    #     else:
    #         classifier_cv = GridSearchCV(pipeline , params[i], scoring=scoring, cv= 4,refit="score", n_jobs=-1, verbose=2)
    #     classifier_cv.fit(Xtrain, ytrain)
    #     Scores.append(classifier_cv.best_score_)
    # print(Scores)
    Scores = []
    scoring = {'score': met.make_scorer(BACC)}
    classifier = Classifiers[0]
    # for i, classifier in enumerate(Classifiers):
    pipeline = imbpipeline(steps = [('sampling', sampling_method(1,1)),('classifier', classifier)])
        # if i == 0:
    classifier_cv = GridSearchCV(pipeline , params[0], scoring=scoring, cv= 4,refit="score", verbose=2)
        # else:
    # classifier_cv = GridSearchCV(pipeline , params[1], scoring=scoring, cv= 4,refit="score", n_jobs=-1, verbose=2)
        # classifier_cv.fit(Xtrain, ytrain)
    classifier_cv.fit(X, Y)
    Scores.append(classifier_cv.best_score_)
    # Scores
    # [0.8820236447035272, 0.9096238273878492, 0.8578112489289657, 0.7784271347421321, 0.8736156059194251, 0.7613127666499254]
    return np.argmax(Scores)

def prediction(Classifiers_best,Classifiers, params):
    # chosen_model = best_model(Classifiers, params) 
    # print(chosen_model)
    chosen_model = 4 # Because KNN yields the best results
    classifier = Classifiers_best[chosen_model]
    X_train, y_train = balance_data(Xtrain, ytrain, 1)
    X_, Y_ = balance_data(X, Y, 1)

    if chosen_model == 0:
        train_labels = tf.keras.utils.to_categorical(y_train, num_classes=3)
        valid_labels = tf.keras.utils.to_categorical(yval, num_classes=3)
        my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
        MLP_train = classifier.fit(X_train, train_labels,validation_data=(Xval, valid_labels), epochs= 40,batch_size=100, callbacks= my_callbacks)
        # Plot(MLP_train)    
        y_pred = classifier.predict(Xval_t)
        y_pred = undo_one_hot_encoding(y_pred)

        y_val_pred = classifier.predict(Xval)
        y_val_pred = undo_one_hot_encoding(y_val_pred)
    else:
        classifier.fit(X_, Y_)
        # classifier.fit(X_train, y_train)
        y_pred = classifier.predict(Xval_t)
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

# print(np.shape(X))
# print(np.shape(X_test))

# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.2)
# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)


    
Classifiers = [KerasClassifier(build_fn=MLP, verbose = False), KNeighborsClassifier(n_jobs=-1), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1),  LogisticRegression(n_jobs=-1)]

best_params = [dict(classifier__n_layers=[ 4], classifier__first_layer_nodes=[128], classifier__last_layer_nodes=[16], classifier__activation_func=['relu'], classifier__epochs=[ 40], classifier__batch_size=[100]),
                dict(classifier__n_neighbors=[4], classifier__algorithm=['auto'], classifier__weights=[ 'distance']), dict(classifier__C =[ 7], classifier__kernel =[ 'rbf'], classifier__gamma =[ .07]),
                dict(classifier__criterion=[ 'gini'], classifier__min_samples_split=[ 5], classifier__splitter=[ 'best']), dict(classifier__criterion=[ 'entropy'], classifier__min_samples_split=[ 2], classifier__n_estimators=[ 150]),
                dict(classifier__penalty =[ 'l2'], classifier__C =[ 2], classifier__solver =[ 'lbfgs'], classifier__max_iter=[1000], classifier__n_jobs=[-1])]

Classifiers_with_best_params = [MLP(n_layers= 4, first_layer_nodes=128, last_layer_nodes=16, activation_func='relu'),
                                KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights= 'distance', n_jobs=-1), SVC(C = 7, kernel = 'rbf' ,gamma = .07),
                                DecisionTreeClassifier(criterion= 'gini', min_samples_split= 5, splitter= 'best'), RandomForestClassifier(criterion= 'entropy', min_samples_split= 2, n_estimators= 150, n_jobs=-1),
                                LogisticRegression(penalty = 'l2', C = 2, solver = 'lbfgs',max_iter=1000, n_jobs=-1)]

X, Xval_t, Y, yval_t = train_test_split(X, Y, test_size=0.3, random_state=2)
# y_sum_predictions = 0
# for i in range(1):
#     print(i)
Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.2)
y_predict = prediction(Classifiers_with_best_params, Classifiers, best_params)
    # y_sum_predictions += tf.keras.utils.to_categorical(y_predict_aux, num_classes=3)

# y_predict = undo_one_hot_encoding(y_sum_predictions)

# 0.7519794654405493
# np.save('Y_Predicted50.npy', y_predict)
# np.save('Y_Predicted4.npy', y_predict)

# y_pred1 = np.load('Y_Predicted50.npy')
# y_pred2 = np.load('Y_Predicted4.npy')

BACC(y_predict, yval_t)

print(np.shape(ytrain))
print(np.shape(yval_t))
print(np.shape(y_predict))