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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


##################################################   Functions   ##################################################


########    Auxiliar function    #######
def categorical_to_binary(y_cat):


    y_bin=[]

    for i in range(len(y_cat)):
        if y_cat[i,0] > y_cat[i,1]:
            y_bin.append(0) 
        elif y_cat[i,0] < y_cat[i,1]:
            y_bin.append(1) 
            
    y_bin = np.ravel(y_bin)
    return y_bin

########    Data Preprocessing     #######
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
        train_labels = categorical_to_binary(train_labels)

    # plt.figure()
    # plt.imshow(Xtrain[500])
    # plt.show()
    return Xtrain, train_labels

def class_weights(y_train):
    cls_wt = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)

    class_weights = {0: cls_wt[0], 1:cls_wt[1]}
    return class_weights



def data_augment(Xtrain):

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2, fill_mode='nearest'),
        ])

    # plt.figure()
    # plt.imshow(Xtrain[500])


    train_dataset = tf.data.Dataset.from_tensor_slices(Xtrain)
    Xtrain = train_dataset.map(lambda x: data_augmentation(x))
    Xtrain = np.array(list(Xtrain))

    # plt.figure()
    # plt.imshow(Xtrain[500])
    # plt.show()

    return Xtrain

########    Evaluation methods     #######
def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)[1]
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)[1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)[1]
    precision = true_positives / (predicted_positives)
    recall = true_positives / (possible_positives)
    f1_val = 2*(precision*recall)/(precision+recall)

    return f1_val

def bacc(cnn_model, X, y_real):
    y_pred = categorical_to_binary(cnn_model.predict(X))
    bacc = BACC(y_real, y_pred)
    return bacc

########    Cost function monitorization     #######
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

########    CNN     #######
def CNN(Xtrain, train_labels, Xval, valid_labels, model, class_weights = None):

    # Model 1
    cnn_model1 = keras.Sequential([
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

        
        


    # Model 2
    cnn_model2 = keras.Sequential([
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


       
    
    if model == 1:
        cnn_model = cnn_model1
    elif model == 2:
        cnn_model = cnn_model2

    
    cnn_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics = f1_score)
    
    my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=15, restore_best_weights=True)]
    cnn_train = cnn_model.fit(Xtrain, train_labels,validation_data=(Xval, valid_labels),class_weight=class_weights, epochs= 50,batch_size=200, callbacks= my_callbacks)


    test_eval = cnn_model.evaluate(Xval, valid_labels, verbose=1)

    # print('Test loss:', test_eval[0])
    # print('Test f1:', test_eval[1])   

    Plots(cnn_train)

    return cnn_model

########    CNN Model evaluation     #######
def model_evaluation(cnn_model, X, y):
    pred_labels = cnn_model.predict(X)
    
    f1 = met.f1_score(y,categorical_to_binary(pred_labels))

    # print(met.confusion_matrix(y,categorical_to_binary(pred_labels)))

    return f1

########    Other classification methods     #######
def other_methods(Xtrain, ytrain, train_labels, Xval, yval, models, scores):
    Xtrain_not_CNN, ytrain_not_CNN = data_balance_generator(Xtrain, train_labels, False)

    # Gaussian Naive Bayes Classifier
    gnb = GaussianNB()
    models.append(gnb)
    pred_y = gnb.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)
    scores.append(met.f1_score(yval,pred_y))
    # print(met.confusion_matrix(yval,pred_y))

    # KNN Classifier
    KNN = KNeighborsClassifier(n_neighbors=51, weights= 'distance')
    models.append(KNN)    
    pred_y = KNN.fit(Xtrain_not_CNN,ytrain_not_CNN).predict(Xval)     
    scores.append(met.f1_score(yval,pred_y))

    # Decision Tree Classifier
    DTC = DecisionTreeClassifier(class_weight = class_weights(ytrain))
    models.append(DTC)
    pred_y = DTC.fit(Xtrain,ytrain).predict(Xval)
    scores.append(met.f1_score(yval,pred_y))

    # Logistic regression
    LogReg = LogisticRegression(C = 0.01, class_weight=class_weights(ytrain))
    models.append(LogReg)
    pred_y = LogReg.fit(Xtrain,ytrain).predict(Xval)
    scores.append(met.f1_score(yval,pred_y))

    # SVC
    svc = SVC(class_weight=class_weights(ytrain),C=3, gamma='scale', kernel='rbf')
    models.append(svc)
    pred_y = svc.fit(Xtrain,ytrain).predict(Xval)
    scores.append(met.f1_score(yval,pred_y))

    return models, scores


########    Preprocessing method to use     #######
def choose_dataset(model):
    l = 150
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
            # pass
            cnn_model = CNN(data_augment(Xtrain_cnn), train_labels, Xval_cnn, valid_labels,model)
            a.append(model_evaluation(cnn_model, Xval_cnn, yval))
            a1.append(bacc(cnn_model, Xval_cnn, yval))
        elif l/5-1<i<2*l/5: # With balanced data
            # pass
            aux_X, aux_y = data_balance_generator(Xtrain_cnn, train_labels)
            cnn_model = CNN(aux_X, aux_y, Xval_cnn, valid_labels,model)
            b.append(model_evaluation(cnn_model, Xval_cnn, yval))
            b1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 2*l/5-1<i<3*l/5: # with original data set
            # pass
            cnn_model = CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,model)
            c.append(model_evaluation(cnn_model, Xval_cnn, yval))
            c1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 3*l/5-1<i<4*l/5: # With data augmentation and balanced data set
            # pass
            aux_X, aux_y = data_balance_generator(Xtrain_cnn, train_labels)
            cnn_model = CNN(data_augment(aux_X), aux_y, Xval_cnn, valid_labels,model)
            d.append(model_evaluation(cnn_model, Xval_cnn, yval))
            d1.append(bacc(cnn_model, Xval_cnn, yval))
        elif 4*l/5-1<i<l: # With re-weighting of classes
            cnn_model = CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,model, class_weights= class_weights(ytrain))
            e.append(model_evaluation(cnn_model, Xval_cnn, yval))
            e1.append(bacc(cnn_model, Xval_cnn, yval))




    print('f1 std:',np.std(a), 'f1 mean:',np.mean(a), 'bacc std:',np.std(a1), 'bacc mean:',np.mean(a1),'max f1:', np.max(a),'min f1:', np.min(a))
    print('f1 std:',np.std(b), 'f1 mean:',np.mean(b), 'bacc std:',np.std(b1), 'bacc mean:',np.mean(b1),'max f1:', np.max(b),'min f1:', np.min(b))
    print('f1 std:',np.std(c), 'f1 mean:',np.mean(c), 'bacc std:',np.std(c1), 'bacc mean:',np.mean(c1),'max f1:', np.max(c),'min f1:', np.min(c))
    print('f1 std:',np.std(d), 'f1 mean:',np.mean(d), 'bacc std:',np.std(d1), 'bacc mean:',np.mean(d1),'max f1:', np.max(d),'min f1:', np.min(d))
    print('f1 std:',np.std(e), 'f1 mean:',np.mean(e), 'bacc std:',np.std(e1), 'bacc mean:',np.mean(e1),'max f1:', np.max(e),'min f1:', np.min(e))

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

    ### 20 iterations ###

    # f1 std: 0.01042932479513175 f1 mean: 0.8057264968840661 bacc std: 0.008598896938537928 bacc mean: 0.8425141792618411 # data augmentation
    # f1 std: 0.008927955038856274 f1 mean: 0.8028322540668462 bacc std: 0.007309293566566375 bacc mean: 0.8408155362536359 # balanced data # top 3 best results
    # f1 std: 0.007801911070372407 f1 mean: 0.8046285695980686 bacc std: 0.00603277575898918 bacc mean: 0.8417473052425283 # original data set # top 3 best results
    # f1 std: 0.009922040979015273 f1 mean: 0.8020404158706482 bacc std: 0.008447145943603306 bacc mean: 0.8400539339117709 # data augmentation and balanced data set
    # f1 std: 0.007474735482207277 f1 mean: 0.8022291009461915 bacc std: 0.006084409747117845 bacc mean: 0.8407527025099386 # re-weighed data # top 3 best results

    ### 30 iterations ###

    # f1 std: 0.00966996612830268 f1 mean: 0.8022678465789693 bacc std: 0.007999945944957315 bacc mean: 0.8402188159605719 max f1: 0.8213991769547324 min f1: 0.7814029363784666# balanced data
    # f1 std: 0.00827838193453924 f1 mean: 0.8045154805829963 bacc std: 0.0068608114661904045 bacc mean: 0.8413744761916574 max f1: 0.8196202531645569 min f1: 0.7791563275434243 # original data set
    # f1 std: 0.010274636787714847 f1 mean: 0.8015368364139019 bacc std: 0.008487780318032446 bacc mean: 0.8396714401189126 max f1: 0.8296178343949044 min f1: 0.7846808510638298 # re-weighed data # best results

    ### 30 iterations ###

    # f1 std: 0.007508302846159263 f1 mean: 0.8306620475532308 bacc std: 0.006394927596692175 bacc mean: 0.8616970821309148 max f1: 0.8442105263157895 min f1: 0.8131868131868132 # data augmentation
    # f1 std: 0.010405802899838579 f1 mean: 0.8284765058890146 bacc std: 0.009200387643840808 bacc mean: 0.8609366039118684 max f1: 0.8518134715025906 min f1: 0.8117913832199545 # balanced data
    # f1 std: 0.00856334517242131 f1 mean: 0.8274182334577067 bacc std: 0.007544861452551712 bacc mean: 0.8589449406806835 max f1: 0.8417721518987342 min f1: 0.8049886621315192 # original data set
    # f1 std: 0.00954929450719991 f1 mean: 0.8299736041026401 bacc std: 0.007889232449550813 bacc mean: 0.8621540928038112 max f1: 0.8495762711864409 min f1: 0.8116545265348595 # data augmentation and balanced data set
    # f1 std: 0.008325136378670762 f1 mean: 0.826220853693954 bacc std: 0.007102912208344387 bacc mean: 0.8589366039118684 max f1: 0.8435814455231931 min f1: 0.8097886540600667 # re-weighed data

    ### 30 iterations ###

    # f1 std: 0.01042932479513175 f1 mean: 0.8306620475532308 bacc std: 0.008598896938537928 bacc mean: 0.8616970821309148 max f1: 0.8442105263157895 min f1: 0.8131868131868132 # data augmentation
    # f1 std: 0.010405802899838579 f1 mean: 0.8284765058890146 bacc std: 0.009200387643840808 bacc mean: 0.8609366039118684 max f1: 0.8518134715025906 min f1: 0.8117913832199545 # balanced data
    # f1 std: 0.00856334517242131 f1 mean: 0.8274182334577067 bacc std: 0.007544861452551712 bacc mean: 0.8589449406806835 max f1: 0.8417721518987342 min f1: 0.8049886621315192 # original data set
    # f1 std: 0.00954929450719991 f1 mean: 0.8299736041026401 bacc std: 0.007889232449550813 bacc mean: 0.8621540928038112 max f1: 0.8495762711864409 min f1: 0.8116545265348595 # data augmentation and balanced data set
    # f1 std: 0.008325136378670762 f1 mean: 0.829220853693954 bacc std: 0.007102912208344387 bacc mean: 0.8589366039118684 max f1: 0.8435814455231931 min f1: 0.8097886540600667 # re-weighed data
##################################################   Main   ##################################################

X = np.load('Xtrain_Classification1.npy')
Y = np.load('ytrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')

#Normalize data   

X = X/255
X_test = X_test/255

# print(np.shape(X))

scores = [0]


Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, shuffle=True)




# One-hot enconding
train_labels = keras.utils.to_categorical(ytrain, num_classes=2)
valid_labels = keras.utils.to_categorical(yval, num_classes=2)

# Reshape
Xtrain_cnn = Xtrain.reshape(-1, 30, 30, 3)
Xval_cnn = Xval.reshape(-1, 30, 30, 3)

# plt.imshow(Xtrain_cnn[504])
# plt.show()
# choose_dataset(2)

# while np.max(scores) < 0.81:
#     #Split data

#     # Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, random_state=2)
#     Xtrain, Xval, ytrain, yval = train_test_split(X, Y, test_size=0.15, shuffle=True)




#One-hot enconding
train_labels = keras.utils.to_categorical(ytrain, num_classes=2)
valid_labels = keras.utils.to_categorical(yval, num_classes=2)

#Reshape
Xtrain_cnn = Xtrain.reshape(-1, 30, 30, 3)
Xval_cnn = Xval.reshape(-1, 30, 30, 3)
#     # choose_dataset(2)

#     # Auxiliar lists
models = []
#     scores = []
# CNN model 1
models.append(CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,1, class_weights= class_weights(ytrain))) 
# scores.append(model_evaluation(models[0], Xval_cnn, yval))

# CNN model 2
models.append(CNN(Xtrain_cnn, train_labels, Xval_cnn, valid_labels,2, class_weights= class_weights(ytrain))) 
#     scores.append(model_evaluation(models[1], Xval_cnn, yval))

#     # Other methods
#     models, scores = other_methods(Xtrain, ytrain, train_labels, Xval, yval, models, scores)

#     print(np.max(scores))



# # Choose best Classifier

# max_f1 = np.argmax(scores)

# print(scores)
# print(max_f1)

# classifier = models[max_f1]

# if max_f1 < 2:
#     X_test = X_test.reshape(-1, 30, 30, 3)
#     y_predict = classifier.predict(X_test)
#     y_predict = categorical_to_binary(y_predict)
# else:
#     y_predict = classifier.fit(X,Y).predict(X_test)

# np.save('Y_Predicted3.npy', y_predict)



# print(np.shape(X_test))
# print(np.shape(y_predict))

# [0.7910798122065726, 0.8296460176991151, 0.5306495882891127, 0.706405693950178, 0.6130434782608696, 0.6577693040991421, 0.7459252157238734] # scores
