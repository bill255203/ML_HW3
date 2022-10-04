# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:28:00 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606. ])
y_data = np.array([ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591. ])

# Function y = b + w1 * x
savepath = './'

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
    fig = plt.figure()
    plt.grid(True)
    plt.plot(x_range,y_range, '-', lw=2, color='red')
    plt.plot(x_data,y_data, 'o', ms=7, color='blue')
    plt.xlim(0,800)
    plt.ylim(0,2000)
    plt.xlabel('x', fontfamily = 'Arial', fontsize = 14)
    plt.ylabel('y', fontfamily = 'Arial', fontsize = 14)
    plt.title('Model (Degree=%d)' %(variable - 1), fontfamily = 'Arial', fontsize = 16)
    plt.savefig(savepath + 'Least_squares_method_' + str(variable - 1) + '.png')
    plt.show(fig)