"""
*******************************************************************************************
Autores: Duarte Silva (ist193243) e João Paiva (ist1105737)
IST     Data: 22/10/2022
Descrição: Classificação Parte 1
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


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)[1]
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)[1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)[1]
    precision = true_positives / (predicted_positives)
    recall = true_positives / (possible_positives)
    f1_val = 2*(precision*recall)/(precision+recall)

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
# X_test = np.load('Xtest_Classification1.npy')

# print(np.count_nonzero(Y==0))
# print(np.count_nonzero(Y==1))
# train_set = X.reshape(-1, 30, 30, 3)
# plt.imshow(train_set[500])
# plt.show()
#### SPLIT DATA   ####

Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.20, random_state=2)
# Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.10, shuffle=True)

####  NORMALIZE DATA   ####

Xtrain = Xtrain/255
Xval = Xval/255


#One-hot enconding
train_labels = keras.utils.to_categorical(ytrain, num_classes=2)
valid_labels = keras.utils.to_categorical(yval, num_classes=2)

#Reshape
Xtrain_cnn = Xtrain.reshape(-1, 30, 30, 3)
Xval_cnn = Xval.reshape(-1, 30, 30, 3)


def data_balance_generator(Xtrain, train_labels, CNN = True):
    # plt.figure()
    # plt.imshow(Xtrain[500])

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

# print(len(train_labels))
# print(np.count_nonzero(train_labels ==  [0., 1.]))
# print(np.count_nonzero(train_labels !=  [0., 1.]))





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

def class_weights(y_train):
    cls_wt = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)

    class_weights = {0: cls_wt[0], 1:cls_wt[1]}
    return class_weights



def data_augment(Xtrain):


    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        # layers.RandomFlip("horizontal_and_vertical", input_shape=(30,30,3)),
        layers.RandomRotation(0.2, fill_mode='nearest'),
        # layers.RandomZoom(0.1)
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
def CNN(Xtrain, train_labels, Xval, valid_labels, model, class_weights = None):

    def model_1(Xtrain, train_labels, Xval, valid_labels, class_weights = None):
        cnn_model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3),1, activation='relu', input_shape=(30, 30, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3),1, activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3),1, activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2,activation='softmax')
        ])

        my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
        cnn_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics = f1_score)


        cnn_train = cnn_model.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels),class_weight=class_weights, epochs= 50,batch_size=200, callbacks= my_callbacks)

        test_eval = cnn_model.evaluate(Xval, valid_labels, verbose=1)

        # Plots(cnn_train)

        return cnn_model


#     cnn_model = keras.Sequential([
#     keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(30, 30, 3)),
#     keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'),
#     keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')
# ])
    def model_2(Xtrain, train_labels, Xval, valid_labels, class_weights = None):
        cnn_model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(30, 30, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.7),
        keras.layers.Dense(2, activation='softmax')
        ])


        my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)]
        cnn_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics = f1_score)


        cnn_train = cnn_model.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels),class_weight=class_weights, epochs= 50,batch_size=200, callbacks= my_callbacks)


        test_eval = cnn_model.evaluate(Xval, valid_labels, verbose=1)

        # Plots(cnn_train)
        return cnn_model
    
    if model == 1:
        cnn_model = model_1(Xtrain, train_labels, Xval, valid_labels, class_weights = None)
    elif model == 2:
        cnn_model = model_2(Xtrain, train_labels, Xval, valid_labels, class_weights = None)

    return cnn_model












#Avaliar modelo
def model_evaluation(cnn_model, X, y):
    pred_labels = cnn_model.predict(X)
    

    print(met.f1_score(y,traduz_vec(pred_labels)))
    print(met.confusion_matrix(y,traduz_vec(pred_labels)))



    # print(f1_score(valid_labels,pred_labels))

    return met.f1_score(y,traduz_vec(pred_labels))

def bacc(cnn_model, X, y_real):
    y_pred = traduz_vec(cnn_model.predict(X))
    bacc = BACC(y_real, y_pred)
    return bacc

def choose_dataset(model):
    l = 5
    a = []
    b = []
    c = []
    d = []
    e = []
    a1 = []
    b1 = []
    c1 = []
    d1 = []
    e1 = []

    for i in range(l):
        print(i)
        if i<l/5: # With data augmentation
            pass
            # cnn_model = CNN(data_augment(Xtrain_cnn), train_labels, Xval_cnn, valid_labels,model)
            # a.append(model_evaluation(cnn_model, Xval_cnn, yval))
            # a1.append(bacc(cnn_model, Xval_cnn, yval))
        elif l/5-1<i<2*l/5: # With balanced data
            aux_X, aux_y = data_balance_generator(Xtrain_cnn, train_labels)
            cnn_model = CNN(aux_X, aux_y, Xval_cnn, valid_labels,model)
            b.append(model_evaluation(cnn_model, Xval_cnn, yval))
            b1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 2*l/5-1<i<3*l/5: # with original data set
            cnn_model = CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,model)
            c.append(model_evaluation(cnn_model, Xval_cnn, yval))
            c1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 3*l/5-1<i<4*l/5: # With data augmentation and balanced data set
            pass
            # aux_X, aux_y = data_balance_generator(Xtrain_cnn, train_labels)
            # cnn_model = CNN(data_augment(aux_X), aux_y, Xval_cnn, valid_labels,model)
            # d.append(model_evaluation(cnn_model, Xval_cnn, yval))
            # d1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 4*l/5-1<i<l: # With re-weighting of classes
            cnn_model = CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,model, class_weights= class_weights(ytrain))
            e.append(model_evaluation(cnn_model, Xval_cnn, yval))
            e1.append(bacc(cnn_model, Xval_cnn, yval))




    # print('f1 std:',np.std(a), 'f1 mean:',np.mean(a), 'bacc std:',np.std(a1), 'bacc mean:',np.mean(a1))
    print('f1 std:',np.std(b), 'f1 mean:',np.mean(b), 'bacc std:',np.std(b1), 'bacc mean:',np.mean(b1),'max f1:', np.max(b),'min f1:', np.min(b))
    print('f1 std:',np.std(c), 'f1 mean:',np.mean(c), 'bacc std:',np.std(c1), 'bacc mean:',np.mean(c1),'max f1:', np.max(c),'min f1:', np.min(c))
    # print('f1 std:',np.std(d), 'f1 mean:',np.mean(d), 'bacc std:',np.std(d1), 'bacc mean:',np.mean(d1))
    print('f1 std:',np.std(e), 'f1 mean:',np.mean(e), 'bacc std:',np.std(e1), 'bacc mean:',np.mean(e1),'max f1:', np.max(e),'min f1:', np.min(e))


# choose_dataset(2)


# plt.figure()
# plt.boxplot(a)
# plt.figure()
# plt.boxplot(b)
# plt.figure()
# plt.boxplot(c)
# plt.figure()
# plt.boxplot(d)
# plt.figure()
# plt.boxplot(e)
# plt.show()

# 0.014810402870845648 0.752717305862203
# 0.010054676105688885 0.7863161712506432
# 0.0296444116535936 0.7803158358300042
# 0.012347847235487494 0.760021928518897
# 0.010845099912714089 0.7826304746877601

# 0.008567417224620033 0.7985125292654413
# 0.006124870971163109 0.7986370070354757
# 0.010937996003127866 0.8040081214211098
# 0.010986082794055175 0.7973686688319737
# 0.007121930135348871 0.8038244610726724


### 20 iterations ###

# f1 std: 0.01042932479513175 f1 mean: 0.8057264968840661 bacc std: 0.008598896938537928 bacc mean: 0.8425141792618411 # data augmentation
# f1 std: 0.008927955038856274 f1 mean: 0.8028322540668462 bacc std: 0.007309293566566375 bacc mean: 0.8408155362536359 # balanced data
# f1 std: 0.007801911070372407 f1 mean: 0.8046285695980686 bacc std: 0.00603277575898918 bacc mean: 0.8417473052425283 # original data set
# f1 std: 0.009922040979015273 f1 mean: 0.8020404158706482 bacc std: 0.008447145943603306 bacc mean: 0.8400539339117709 # data augmentation and balanced data set
# f1 std: 0.007474735482207277 f1 mean: 0.8022291009461915 bacc std: 0.006084409747117845 bacc mean: 0.8407527025099386 # re-weighed data

### 30 iterations ###

# f1 std: 0.00966996612830268 f1 mean: 0.8022678465789693 bacc std: 0.007999945944957315 bacc mean: 0.8402188159605719 max f1: 0.8213991769547324 min f1: 0.7814029363784666# balanced data
# f1 std: 0.00827838193453924 f1 mean: 0.8045154805829963 bacc std: 0.0068608114661904045 bacc mean: 0.8413744761916574 max f1: 0.8196202531645569 min f1: 0.7791563275434243 # original data set
# f1 std: 0.010274636787714847 f1 mean: 0.8015368364139019 bacc std: 0.008487780318032446 bacc mean: 0.8396714401189126 max f1: 0.8296178343949044 min f1: 0.7846808510638298 # re-weighed data