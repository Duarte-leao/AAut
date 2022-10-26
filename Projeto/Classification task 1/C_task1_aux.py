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
import tensorflow_addons as tfa
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

# #One-hot enconding
train_labels = keras.utils.to_categorical(ytrain, num_classes=2)
valid_labels = keras.utils.to_categorical(yval, num_classes=2)
# test_labels = keras.utils.to_categorical(ytest, num_classes=2)
# print(test_labels)
# print(ytest)

# training_generator = BalancedBatchGenerator(
#     Xtrain, train_labels, sampler=RandomOverSampler(), batch_size=1, random_state=42)

# training_generator = balanced_batch_generator(
#     Xtrain, train_labels, sampler=RandomOverSampler(), batch_size=1, random_state=42)

# print((training_generator[0]))


def data_balance_generator(Xtrain, train_labels):
    # plt.figure()
    # plt.imshow(Xtrain[500])

    Xtrain = Xtrain.reshape(-1, 2700)
    training_generator = balanced_batch_generator(
        Xtrain, train_labels, sampler=RandomOverSampler(sampling_strategy=1), batch_size=1)
    # training_generator = balanced_batch_generator(
    #     Xtrain, train_labels, batch_size=1)

    Xtrain = []
    train_labels = []
    for i, el in enumerate(training_generator[0]): 
        # print(i, el)
        # print(el[0])
        Xtrain.append(el[0])
        train_labels.append(el[1])
        if i == training_generator[1]:
            break
    Xtrain = np.vstack(Xtrain)
    train_labels = np.vstack(train_labels)
    Xtrain = Xtrain.reshape(-1, 30, 30, 3)

    # plt.figure()
    # plt.imshow(Xtrain[500])
    # plt.show()
    return Xtrain, train_labels

# print(len(train_labels))
# print(np.count_nonzero(train_labels ==  [0., 1.]))
# print(np.count_nonzero(train_labels !=  [0., 1.]))

#Reshape
Xtrain = Xtrain.reshape(-1, 30, 30, 3)
Xval = Xval.reshape(-1, 30, 30, 3)
# Xtest = Xtest.reshape(-1, 30, 30, 3)

# print(Xtrain)
########    CNN     #######


# #### Modelo 1 ####


# cnn_model1 = keras.Sequential()
# cnn_model1.add(keras.layers.Conv2D(32, (3, 3),1, activation='relu', input_shape=(30, 30, 3)))
# cnn_model1.add(keras.layers.MaxPooling2D((2, 2)))
# cnn_model1.add(keras.layers.Conv2D(64, (3, 3),1, activation='relu'))
# cnn_model1.add(keras.layers.MaxPooling2D((2, 2)))
# cnn_model1.add(keras.layers.Flatten())
# cnn_model1.add(keras.layers.Dense(64, activation='relu'))
# cnn_model1.add(keras.layers.Dense(2,activation='softmax'))




# my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=True)]
# cnn_model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics = f1_score)
# cnn1_train = cnn_model1.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels), epochs= 30,batch_size=200, callbacks= my_callbacks)
# # test_eval = cnn_model1.evaluate(Xval, valid_labels, verbose=1)
# test_eval = cnn_model1.evaluate(Xtest, test_labels, verbose=1)


# #Imprimir resultados 
# print('Test loss:', test_eval[0])
# print('Test f1:', test_eval[1])
# Plots(cnn1_train)

#### Modelo 2 ####





def data_augment(Xtrain):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        # layers.RandomFlip("horizontal_and_vertical", input_shape=(30,30,3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1)
        ])

    # plt.figure()
    # plt.imshow(Xtrain[500])


    train_dataset = tf.data.Dataset.from_tensor_slices(Xtrain)
    Xtrain = train_dataset.map(lambda x: data_augmentation(x))
    Xtrain = np.array(list(Xtrain))
    return Xtrain
    # train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, train_labels))
    # train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    # train_dataset = train_dataset.batch(200).map(lambda x, y: (data_augmentation(x), y))


# plt.figure()
# plt.imshow(Xtrain[500])
# plt.show()

# Xtrain = data_augment(Xtrain)
# Xtrain, train_labels = data_balance_generator(Xtrain, train_labels)
# Xtrain = data_augment(Xtrain)
def CNN(Xtrain, train_labels, Xval, valid_labels):
    cnn_model1 = keras.Sequential()
    # cnn_model1.add(data_augmentation)
    cnn_model1.add(keras.layers.Conv2D(16, (3, 3),1, activation='relu', input_shape=(30, 30, 3)))
    cnn_model1.add(keras.layers.MaxPooling2D((2, 2)))
    cnn_model1.add(keras.layers.Conv2D(32, (3, 3),1, activation='relu'))
    cnn_model1.add(keras.layers.MaxPooling2D((2, 2)))
    cnn_model1.add(keras.layers.Conv2D(64, (3, 3),1, activation='relu'))
    cnn_model1.add(keras.layers.MaxPooling2D((2, 2)))
    # cnn_model1.add(layers.Dropout(0.2))

    cnn_model1.add(keras.layers.Flatten())
    cnn_model1.add(keras.layers.Dense(128, activation='relu'))
    cnn_model1.add(keras.layers.Dense(2,activation='softmax'))
    # cnn_model1.add(keras.layers.Dense(1,activation='softmax'))

    my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=True)]
    cnn_model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics = f1_score)
    # cnn_model1.compile(optimizer="adam", loss='binary_crossentropy', metrics = f1_score)

    cnn1_train = cnn_model1.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels), epochs= 30,batch_size=200, callbacks= my_callbacks)
    # cnn1_train = cnn_model1.fit(Xtrain, ytrain,validation_data=(Xval, yval), epochs= 30,batch_size=200, callbacks= my_callbacks)

    # cnn1_train = cnn_model1.fit(train_dataset,validation_data=(Xval, valid_labels), epochs= 30,batch_size=200, callbacks= my_callbacks)

    test_eval = cnn_model1.evaluate(Xval, valid_labels, verbose=1)
    # test_eval = cnn_model1.evaluate(Xval, yval, verbose=1)

    # test_eval = cnn_model1.evaluate(Xtest, test_labels, verbose=1)
    return cnn_model1







# usar a função predict

#Imprimir resultados 
# print('Test loss:', test_eval[0])
# print('Test f1:', test_eval[1])
# Plots(cnn1_train)

#Avaliar modelo
def model_evaluation(cnn_model, X, y):
    pred_labels = cnn_model.predict(X)
    
    # pred_y=traduz_vec(pred_labels)

    print(met.f1_score(y,traduz_vec(pred_labels)))
    print(met.confusion_matrix(y,traduz_vec(pred_labels)))
    # print(met.f1_score(y,pred_labels))

    # print(f1_score(y.reshape(-1,1).astype('float32'),pred_labels))
    # print(pred_labels)
    # print(valid_labels)
    print(f1_score(valid_labels,pred_labels))

    return met.f1_score(y,traduz_vec(pred_labels))

l = 36
a = []
b = []
c = []
d = []
for i in range(l):
    print(i)
    if i<l/4: # With data augmentation
        cnn_model = CNN(data_augment(Xtrain), train_labels, Xval, valid_labels)
        a.append(model_evaluation(cnn_model, Xval, yval))
    elif l/4-1<i<2*l/4: # With balanced data
        aux_X, aux_y = data_balance_generator(Xtrain, train_labels)
        cnn_model = CNN(aux_X, aux_y, Xval, valid_labels)
        b.append(model_evaluation(cnn_model, Xval, yval))
    elif 2*l/4-1<i<3*l/4: # with original data set
        cnn_model = CNN(Xtrain, train_labels, Xval, valid_labels)
        c.append(model_evaluation(cnn_model, Xval, yval))
    elif 3*l/4-1<i<l: # With data augmentation and balanced data set
        aux_X, aux_y = data_balance_generator(Xtrain, train_labels)
        cnn_model = CNN(data_augment(aux_X), aux_y, Xval, valid_labels)
        d.append(model_evaluation(cnn_model, Xval, yval))
    # elif 3*l/4-1<i<l: # With data augmentation and balanced data set
    #     aux_X, aux_y = data_balance_generator(Xtrain, train_labels)
    #     cnn_model = CNN(data_augment(aux_X), aux_y, Xval, valid_labels)
    #     d.append(model_evaluation(cnn_model, Xval, valid_labels))


print(np.std(a), np.mean(a))
print(np.std(b), np.mean(b))
print(np.std(c), np.mean(c))
print(np.std(d), np.mean(d))



##############Logistic Regression###################

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(Xtrain,ytrain)
model.predict(Xval)
model.score(Xval,yval)