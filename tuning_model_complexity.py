# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:12:46 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([ 338., 333., 328., 226., 25., 179., 60., 208.])
y_data = np.array([ 640., 633., 619., 428., 27., 193., 66., 226.])

x_valid = np.array([207,606.])
y_valid = np.array([393,1591.])

# Function y = b + w1 * x
savepath = './'

loss_history = []

for var in range(6):
    variable = var + 2
    
    X = np.zeros((len(x_data),variable))
    Y = y_data.reshape(len(y_data),1)
    
    for i in range(variable):
        x = x_data.reshape(len(x_data))
        X[:,i] = 1 * (x**i)
    
    X_t = X.transpose()
    matrix = np.dot(X_t,X)
    matrix_inverse = np.linalg.inv(matrix)
    para = np.dot(matrix_inverse,np.dot(X_t,Y))
    
    x_range = np.arange(0,800,0.1)
    y_range = np.zeros(len(x_range))
    for i in range(len(para)):
        y_range = y_range + para[i] * (x_range**i)
    
    y_pred = np.zeros(len(x_valid))
    for i in range(len(para)):
        y_pred = y_pred + para[i] * (x_valid**i)
    loss = 0
    for i in range(len(x_valid)):
        loss = loss + (y_valid[i] - y_pred[i])**2
    loss = loss/len(x_valid)
    loss_history.append(loss)
    
    fig = plt.figure()
    plt.grid(True)
    plt.plot(x_range,y_range, '-', lw=2, color='red')
    plt.plot(x_data,y_data, 'o', ms=7, color='blue')
    plt.plot(x_valid,y_valid, 'o', ms=7, color='blue')
    plt.xlim(0,800)
    plt.ylim(0,2000)
    plt.xlabel('x', fontfamily = 'Arial', fontsize = 14)
    plt.ylabel('y', fontfamily = 'Arial', fontsize = 14)
    plt.title('Model (Degree=%d)' %(variable - 1), fontfamily = 'Arial', fontsize = 16)
    plt.savefig(savepath + 'Tuning_Model_Complexity' + str(variable - 1) + '.png')
    plt.show(fig)

times = np.arange(0,3,1)
loss_matrix = np.zeros(3)
for i in range(3):
    loss_matrix[i] = loss_history[i]
fig1 = plt.figure()
plt.grid(True)
plt.plot(times,loss_matrix, 'o-', lw=2, ms=10, color='red')
plt.xlim(0,2)
plt.ylim(40000,1200000)
plt.xticks([0,1,2],['1','2','3'])
plt.xlabel('Model degree', fontfamily = 'Arial', fontsize = 14)
plt.ylabel('Error', fontfamily = 'Arial', fontsize = 14)
plt.title('Error vs. Model degree', fontfamily = 'Arial', fontsize = 16)
plt.savefig(savepath + 'Comparsion' + '.png')
plt.show(fig1)