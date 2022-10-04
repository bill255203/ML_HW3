# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:01:39 2020

@author: User
"""


import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606. ])
y_data = np.array([ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591. ])
# y_data = b + w1 * x_data

x = np.arange(-200,-100,1) #bias
y = np.arange(-5,5,0.1) #weight1
Z = np.zeros((len(x),len(y)))
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w1 = y[j]
        Z[j][i] = 0
        for k in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[k] - b - w1*x_data[k])**2
        Z[j][i] = Z[j][i]/len(x_data)

b = -120 # initial b
w1 = -4 # initital w
lr = 1 # learning rate
iteration = 100000

# Store initial value for plotting
b_history = [b]
w1_history = [w1]

lr_b = 0
lr_w1 = 0

# Iterations
for i in range(iteration):
    b_grad = 0.0
    w1_grad = 0.0
    for j in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[j] - (b + w1 * x_data[j]))*1.0
        w1_grad = w1_grad - 2.0*(y_data[j] - (b + w1 * x_data[j]))*x_data[j]
    
    lr_b = lr_b + b_grad **2
    lr_w1 = lr_w1 + w1_grad **2
    
    # Updata parameters
    b = b - lr/np.sqrt(lr_b) * b_grad
    w1 = w1 - lr/np.sqrt(lr_w1) * w1_grad
    
    # Store parameters for plotting
    b_history.append(b)
    w1_history.append(w1)
    
    print('Iteration %d Done' %(i+1))

fig1 = plt.figure()
plt.contourf(X,Y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67], 'x', ms=12, markeredgewidth=3, color='orange')    
plt.plot(b_history, w1_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel('b', fontsize = 12)
plt.ylabel('w1', fontsize = 12)
plt.title('Gradient descent', fontsize = 16)
plt.show(fig1)

last_b = b_history[iteration]
last_w1 = w1_history[iteration]
x_range = np.arange(0,800,0.1)
y_range = last_b + last_w1 * x_range
fig2 = plt.figure()
plt.grid(True)
plt.plot(x_range,y_range, '-', lw=2, color='red')
plt.plot(x_data,y_data, 'o', ms=7, color='blue')
plt.xlim(0,800)
plt.ylim(0,2000)
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.title('Linear model', fontsize = 16)
plt.show(fig2)