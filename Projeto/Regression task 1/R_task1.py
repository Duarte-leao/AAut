import numpy as np
from sklearn import linear_model as lm


X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')

Xtrain = X[:95,:]
Ytrain = Y[:95,:]

Xval = X[95:,:]
Yval = Y[95:,:]

reg = lm.LinearRegression()
reg.fit(Xtrain,Ytrain)

Ypred = reg.predict(Xval)

SSE =  np.sum((Yval-Ypred)**2)
print(SSE)



""" print(reg.coef_)
print(reg.score(Xtrain,Ytrain)) """