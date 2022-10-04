# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:06:04 2021

@author: User
"""


import numpy as np
import random
import matplotlib.pyplot as plt

#%%
# Function y = b + w1 * x + w2 * (x**2)

def make_data(N,err=1,rseed=1):
    rng=np.random.RandomState(rseed)
    x = rng.rand(N,1)**2
    y = 10-1/(x.ravel()+0.1)
    if err>0:
        y+=err*rng.randn(N)
    
    return x,y

x,y = make_data(100)

x = x.reshape(100)
print(x)
#%%
def LossFunction(x,y,b,w1,w2):
    loss = 0.0
    y_pred = b + w1 * x + w2 * (x*x)
    for i in range(len(y_pred)):
        loss = loss + (y[i] - y_pred[i])**2
    loss = loss/len(y_pred)
    
    return loss

#%%
# Initial parameter
b = random.random()
w1 = random.random()
w2 = random.random()
lr = 0.0001

b_history = [b]
w1_history = [w1]
w2_history = [w2]
loss_history = []

iteration = 100

for i in range(iteration):
    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    for j in range(len(x)):
        y_pred = b + w1 * x[j] + w2 * (x[j]**2)
        b_grad = b_grad - 2 * (y[j] - y_pred) * 1
        w1_grad = w1_grad - 2 * (y[j] - y_pred) * x[j]
        w2_grad = w2_grad - 2 * (y[j] - y_pred) * (x[j]**2)
    
    b = b - lr * b_grad
    w1 = w1 - lr * w1_grad
    w2 = w2 - lr * w2_grad
    
    b_history.append(b)
    w1_history.append(w1)
    w2_history.append(w2)
    
    loss = LossFunction(x,y,b,w1,w2)
    loss_history.append(loss)
    
    print('=== Iteration: %d ===' %(i+1))
    print('Loss: %.4f' %loss)

#%%
# Plot
x_line = np.arange(0,1,0.001)
y_line = b + w1 * x_line + w2 * (x_line**2)
fig1 = plt.figure()
plt.grid(True)
plt.scatter(x, y, color = 'blue')
plt.plot(x_line, y_line, '-', lw = 3, color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-0.2,1.2)
plt.ylim(-2,12)
plt.title('Polynomial Model')
plt.show()

interval = np.arange(0,iteration,1)
fig2 = plt.figure()
plt.grid(True)
plt.plot(interval, loss_history, 'o-', ms = 3, lw = 1, color = 'orange')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xlim(0,iteration)
plt.ylim(0,40)
plt.title('Loss per iteration')
plt.show()