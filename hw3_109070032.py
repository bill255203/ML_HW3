# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:38:15 2022

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from math import sqrt,exp,pi
from numpy.linalg import inv, det
import random
import sklearn.model_selection
import math
from scipy import stats


#QUESTION 1 HERE
filename='Real_estate.csv'
data = pd.read_csv(filename)

train = data.sample(frac=0.6, random_state=25)
test = data.drop(train.index)
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)
#print(train)
#print(test)

#QUESTION 2 HERE
X2 = list(train['X2 house age'])
X3 = list(train['X3 distance to the nearest MRT station'])
X4 = list(train['X4 number of convenience stores'])
Y = list(train['Y house price of unit area'])

x2 = list(test['X2 house age'])
x3 = list(test['X3 distance to the nearest MRT station'])
x4 = list(test['X4 number of convenience stores'])
y = list(test['Y house price of unit area'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X2, Y, s=5, c='b', label='train')
ax1.scatter(x2, y, s=5, c='r', label='test')
plt.xlabel("X2 house age")
plt.ylabel("Y house price of unit area")
plt.legend(loc='upper right');
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X3, Y, s=5, c='b', label='train')
ax1.scatter(x3, y, s=5, c='r', label='test')
plt.xlabel("X3 distance to the nearest MRT station")
plt.ylabel("Y house price of unit area")
plt.legend(loc='upper right');
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X4, Y, s=5, c='b', label='train')
ax1.scatter(x4, y, s=5, c='r', label='test')
plt.xlabel("X4 number of convenience stores")
plt.ylabel("Y house price of unit area")
plt.legend(loc='upper right');
plt.show()

#%%
#because the numbers of x2 and x3 are too big 
#so we divide the numbers to make them close to 0~10
X2 = np.array(X2)/10
X3 = np.array(X3)/100
X4 = np.array(X4)
Y = np.array(Y)

x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
y = np.array(y)

#QUESTION 3 HERE
def LossFunction(X2,X3,X4,Y,b,w1,w2,w3):
    loss = 0.0
    y_pred = b + w1 * X2 + w2 * X3 + w3 * X4
    for i in range(len(y_pred)):
        loss = loss + (Y[i] - y_pred[i])**2
    loss = loss/len(y_pred)
    
    return loss
#QUESTION 4 HERE
b = random.random()
w1 = random.random()
w2 = random.random()
w3 = random.random()
lr = 0.00001

b_history = [b]
w1_history = [w1]
w2_history = [w2]
w3_history = [w3]
loss_history = []

iteration = 500

for i in range(iteration):
    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    w3_grad = 0.0
    for j in range(len(X2)):
        y_pred = b + w1 * X2[j] + w2 * X3[j] + w3 * X4[j]
        b_grad = b_grad - 2 * (Y[j] - y_pred) * 1
        w1_grad = w1_grad - 2 * (Y[j] - y_pred) * X2[j]
        w2_grad = w2_grad - 2 * (Y[j] - y_pred) * X3[j]
        w3_grad = w2_grad - 2 * (Y[j] - y_pred) * X4[j]
                
    b = b - lr * b_grad
    w1 = w1 - lr * w1_grad
    w2 = w2 - lr * w2_grad
    w3 = w3 - lr * w3_grad
    
    b_history.append(b)
    w1_history.append(w1)
    w2_history.append(w2)
    w3_history.append(w3)
    
    loss = LossFunction(X2,X3,X4,Y,b,w1,w2,w3)
    loss_history.append(loss)
    
    if(i+1)%50 == 0:
        print('=== Iteration: %d ===' %(i+1))
        print('Loss: %.4f' %loss)
    #print b, w1, w2, w3 at last iteration
    if(i+1)%500 == 0:
        print(b, w1/10)
        print(w2/100, w3)  
#%%
#QUESTION 5.1 HERE
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
Y = np.array(Y)

x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
y = np.array(y)

X = np.zeros((len(X2), 4))
Y = Y.reshape(len(Y),1)

X[:,0]=1
X2 = X2.reshape(len(X2))
X3 = X3.reshape(len(X3))
X4 = X4.reshape(len(X4))
X[:,1] = X2
X[:,2] = X3
X[:,3] = X4

X_t = X.transpose()
matrix = np.dot(X_t,X)
matrix_inverse = np.linalg.inv(matrix)
para = np.dot(matrix_inverse,np.dot(X_t,Y))

y_pred = np.zeros(len(x2))
y_pred = y_pred + para[0]
y_pred = y_pred + para[1] * x2
y_pred = y_pred + para[2] * x3
y_pred = y_pred + para[3] * x4

loss = 0
for i in range(len(x2)):
    loss = loss + (y[i] - y_pred[i])**2
loss = loss/len(x2)
print(loss)

y_sum = 0
for i in range(len(y)):
    y_sum+=y[i]
y_average = y_sum/len(y)

y_var = 0
for i in range(len(y)):
    y_var = y_var + (y[i] - y_average)**2
print(y_var)

R_sq = 1-loss/y_var
print(R_sq)

#REPORT 1 HERE
x_range = np.arange(0,45,0.1)
y_range = np.zeros(len(x_range))
y_range = y_range + para[0]
y_range = y_range + para[1] * x_range
y_range = y_range + para[2] * x_range
y_range = y_range + para[3] * x_range

fig = plt.figure()
plt.grid(True)
plt.plot(x_range,y_range, '-', lw=2, color='black')
plt.plot(X2,Y, 'o', ms=2, color='blue')
plt.plot(x2,y, 'o', ms=2, color='red')
plt.xlim(0,45)
plt.ylim(0,120)
plt.xlabel('x2', fontfamily = 'Arial', fontsize = 14)
plt.ylabel('y', fontfamily = 'Arial', fontsize = 14)
plt.show(fig)

#REPORT 2 HERE
x_range = np.arange(0,6500,0.1)
y_range = np.zeros(len(x_range))
y_range = y_range + para[0]
y_range = y_range + para[1] * x_range
y_range = y_range + para[2] * x_range
y_range = y_range + para[3] * x_range

fig = plt.figure()
plt.grid(True)
plt.plot(x_range,y_range, '-', lw=2, color='black')
plt.plot(X3,Y, 'o', ms=2, color='blue')
plt.plot(x3,y, 'o', ms=2, color='red')
plt.xlim(0,6500)
plt.ylim(0,120)
plt.xlabel('x3', fontfamily = 'Arial', fontsize = 14)
plt.ylabel('y', fontfamily = 'Arial', fontsize = 14)
plt.show(fig)


#REPORT 3 HERE
x_range = np.arange(0,11,0.1)
y_range = np.zeros(len(x_range))
y_range = y_range + para[0]
y_range = y_range + para[1] * x_range
y_range = y_range + para[2] * x_range
y_range = y_range + para[3] * x_range

fig = plt.figure()
plt.grid(True)
plt.plot(x_range,y_range, '-', lw=2, color='black')
plt.plot(X4,Y, 'o', ms=2, color='blue')
plt.plot(x4,y, 'o', ms=2, color='red')
plt.xlim(0,11)
plt.ylim(0,120)
plt.xlabel('x4', fontfamily = 'Arial', fontsize = 14)
plt.ylabel('y', fontfamily = 'Arial', fontsize = 14)
plt.show(fig)

#%%
#QUESTION 5.2 HERE
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
Y = np.array(Y)

x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
y = np.array(y)

X = np.zeros((len(X2), 5))
Y = Y.reshape(len(Y),1)

X[:,0]=1
X2 = X2.reshape(len(X2))
X3 = X3.reshape(len(X3))
X4 = X4.reshape(len(X4))
X[:,1] = X2
X[:,2] = X2**2
X[:,3] = X3
X[:,4] = X4

X_t = X.transpose()
matrix = np.dot(X_t,X)
matrix_inverse = np.linalg.inv(matrix)
para = np.dot(matrix_inverse,np.dot(X_t,Y))

y_pred = np.zeros(len(x2))
y_pred = y_pred + para[0]
y_pred = y_pred + para[1] * x2
y_pred = y_pred + para[2] * (x2**2)
y_pred = y_pred + para[3] * x3
y_pred = y_pred + para[4] * x4

loss = 0
for i in range(len(x2)):
    loss = loss + (y[i] - y_pred[i])**2
loss = loss/len(x2)
print(loss)

y_sum = 0
for i in range(len(y)):
    y_sum+=y[i]
y_average = y_sum/len(y)

y_var = 0
for i in range(len(y)):
    y_var = y_var + (y[i] - y_average)**2
print(y_var)

R_sq = 1-loss/y_var
print(R_sq)
#%%
#QUESTION 5.3 HERE
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
Y = np.array(Y)

x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
y = np.array(y)

X = np.zeros((len(X2), 5))
Y = Y.reshape(len(Y),1)

X[:,0]=1
X2 = X2.reshape(len(X2))
X3 = X3.reshape(len(X3))
X4 = X4.reshape(len(X4))
X[:,1] = X2
X[:,2] = X3
X[:,3] = X3**2
X[:,4] = X4

X_t = X.transpose()
matrix = np.dot(X_t,X)
matrix_inverse = np.linalg.inv(matrix)
para = np.dot(matrix_inverse,np.dot(X_t,Y))

y_pred = np.zeros(len(x2))
y_pred = y_pred + para[0]
y_pred = y_pred + para[1] * x2
y_pred = y_pred + para[2] * x3
y_pred = y_pred + para[3] * (x3**2)
y_pred = y_pred + para[4] * x4

loss = 0
for i in range(len(x2)):
    loss = loss + (y[i] - y_pred[i])**2
loss = loss/len(x2)
print(loss)

y_sum = 0
for i in range(len(y)):
    y_sum+=y[i]
y_average = y_sum/len(y)

y_var = 0
for i in range(len(y)):
    y_var = y_var + (y[i] - y_average)**2
print(y_var)

R_sq = 1-loss/y_var
print(R_sq)

