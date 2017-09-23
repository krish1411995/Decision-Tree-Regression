#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:46:10 2017

@author: krishmehta
"""

import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

number_of_samples = 200
x = np.linspace(-np.pi, np.pi, number_of_samples)
y = 0.5*x+np.sin(x)+np.random.random(x.shape)
plt.scatter(x,y,color='black')
plt.xlabel('value of x-input')
plt.ylabel('value of y-target')
plt.show()

random_indices = np.random.permutation(number_of_samples)
#Training set
x_train = x[random_indices[:75]]
y_train = y[random_indices[:75]]
#Validation set
x_val = x[random_indices[75:85]]
y_val = y[random_indices[75:85]]
#Test set
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

maximum_depth_of_tree = 10
train_err_arr = []
val_err_arr = []
test_err_arr = []

for depth in maximum_depth_of_tree:
    
    model = tree.DecisionTreeRegressor(max_depth=depth)
    #sklearn takes the inputs as matrices.
    x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
    y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))
    #Fit the line to the training data
    model.fit(x_train_for_line_fitting, y_train_for_line_fitting)
    #Plot the line
    plt.figure()
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x.reshape((len(x),1)),model.predict(x.reshape((len(x),1))),color='blue')
    plt.xlabel('x-input feature')
    plt.ylabel('y-target values')
    plt.title('maximum depth of tree='+str(depth))
    plt.show()
    #used to get the average error
    mean_train_error = np.mean( (y_train - model.predict(x_train.reshape(len(x_train),1)))**2 )
    mean_val_error = np.mean( (y_val - model.predict(x_val.reshape(len(x_val),1)))**2 )
    mean_test_error = np.mean( (y_test - model.predict(x_test.reshape(len(x_test),1)))**2 )
    # appended to show it in a graph to compare
    train_err_arr.append(mean_train_error)
    val_err_arr.append(mean_val_error)
    test_err_arr.append(mean_test_error)
    
plt.figure()
plt.plot(train_err_arr,c='red')
plt.plot(val_err_arr,c='blue')
plt.plot(test_err_arr,c='green')
plt.legend(['Training error', 'Validation error', 'Test error'])
plt.title('Variation of error with maximum depth of tree')
plt.show()