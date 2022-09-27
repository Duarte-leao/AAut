from re import X
import numpy as np
from sklearn import linear_model as lm


X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')

Xtrain = X[:80,:]
Ytrain = Y[:80,:]

Xval = X[80:,:]
Yval = Y[80:,:]

reg = lm.LinearRegression()
reg.fit(Xtrain,Ytrain)

Ypred = reg.predict(Xval)


print(reg.coef_)
print(reg.score(Xtrain,Ytrain))