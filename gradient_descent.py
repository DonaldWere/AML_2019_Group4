# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:24:55 2019

@author: Group_4
"""

# %% Setting up the session
#Importing packages used in the code
import numpy as np

from  matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %% Setting up the class
class GradientDescent:
    """
    This class contains the Plain-vanilla gradient descent and its two variants,
    i.e Momentum Gradient Descent and Nestrov's Accelerated Gradient Descent
    """
    def __init__(self, fn_loss, fn_grad1, fn_grad2):
        self.fn_loss = fn_loss   #Loss function
        self.fn_grad1 = fn_grad1 #Partial derivative with respect to x
        self.fn_grad2 = fn_grad2 #Partial derivative with respect to y

# %% Setting up the Plain vanilla gradient descent
    def plain_vanilla(self, x_init, y_init, max_iter, eta, tol):
        """The algorithm adjusts the x and y till the function gets to its minimum
        The values of x and y are adjusted solely based on gradient at a given point"""
        # Initialise lists to store results
        loss_path = []
        x_path = []
        y_path = []
        n_iter = []
        # Initializing the original values of x, y, f(x,y) and n_iter
        x_value = x_init
        y_value = y_init
        num_iters = 0
        # Storing these initial values to the lists
        x_path.append(x_value)
        y_path.append(y_value)
        n_iter.append(num_iters)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        # Performing the loop for determinimg the minimum of the function
        for i in range(max_iter):
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                #Loops till the partial derivatives are zero or function diverges
                break
            num_iters = num_iters + 1
            n_iter.append(num_iters)
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            #if the gradient is not zero, the x and y values are adjusted
            # adjustment= step_size(eta)*respective gradient'''
            x_value += -eta * gradient_x
            x_path.append(x_value)
            y_value += -eta * gradient_y
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)
        #Output when running the algorthm
        if np.isnan(gradient_x) or np.isnan(gradient_y):
            print('Diverged at step size= {} after {} steps'.format(np.round(eta, 4), num_iters))
        elif np.abs(gradient_x) > tol or np.abs(gradient_y) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps, step size= {}. Loss fn= {} at (x, y) = {}'
                  .format(num_iters, np.round(eta, 4), np.round(loss_this, 4),
                          (np.round(x_value, 4), np.round(y_value, 4))))
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        self.loss_fn_min = loss_this
        self.x_at_min = x_value
        self.y_at_min = y_value
        self.n_iter = n_iter
        self.num_iters = num_iters

# %% Setting up the Momentum Gradient Descent
    def momentum(self, x_init, y_init, max_iter, eta, tol, alpha):
        """Momentum gradient descent offers a way of making larger steps to
        ensure the function converges faster.
        This is achieved by adjusting the step size under plain-vanilla to
        include a proportion of the last step size."""
        #Initialise lists to score the output of the algorithm
        loss_path = []
        x_path = []
        y_path = []
        n_iter = []
         # Initializing the original values of x, y, f(x,y) and n_iter
        x_value = x_init
        y_value = y_init
        num_iters = 0
        # Storing these initial values to the lists
        x_path.append(x_value)
        y_path.append(y_value)
        n_iter.append(num_iters)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        # Additional parameters. This refer to the last step taken by x_value and y_value
        x_step = 0
        y_step = 0
        for i in range(max_iter):
            gradient_y = self.fn_grad2(x_value, y_value)
            gradient_x = self.fn_grad1(x_value, y_value)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                #Loops till the partial derivatives are either zero or function diverges
                break
            num_iters = num_iters + 1
            n_iter.append(num_iters)
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            #if the gradient is not zero, the x and y values are adjusted by:
            # 1. The step_size(eta)*respective gradient
            # 2. An additional adjustment = last_step size*alpha(alpha set at 0.9)
            x_step = alpha * x_step + eta * gradient_x
            x_value += -x_step
            x_path.append(x_value)
            y_step = alpha * y_step + eta * gradient_y
            y_value += -y_step
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)
        #Output when running the algorthm
        if np.isnan(gradient_x) or np.isnan(gradient_y):
            print('Diverged at step size= {} after {} steps'.format(np.round(eta, 4), num_iters))
        elif np.abs(gradient_x) > tol or np.abs(gradient_y) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps, step size= {}. Loss fn= {} at (x, y) = {}'
                  .format(num_iters, np.round(eta, 4), np.round(loss_this, 4),
                          (np.round(x_value, 4), np.round(y_value, 4))))
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        self.loss_fn_min = loss_this
        self.x_at_min = x_value
        self.y_at_min = y_value
        self.n_iter = n_iter
        self.num_iters = num_iters

# %% Setting up Nestrov's Accelerated Gradient Descent
    def nag(self, x_init, y_init, max_iter, eta, tol, alpha):
        """Nestrov's accelerated gradient descent is similar to Momentum gradient descent.
        It however offers a way of making an informed step size.
        This is achieved by calculating a forward looking gradient
        (gradient calculated at an approximate future position)"""
        # initialise lists to score the path of {x,y} and the path of the loss functions
        x_value = x_init
        y_value = y_init
        num_iters = 0
        loss_path = []
        x_path = []
        y_path = []
        n_iter = []
        x_path.append(x_value)
        y_path.append(y_value)
        n_iter.append(num_iters)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        x_step = 0
        y_step = 0
        for i in range(max_iter):
            # i starts from 0 so add 1
            gamma = 1 - 3 / (i + 1 + 5)
            #Gamma represents the weight given to the last step size when
            #determinimg the approximate future position.
            #The gradient is calculated at the approximate future x and y.
            gradient_x = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            gradient_y = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                break
            num_iters = num_iters + 1
            n_iter.append(num_iters)
            gradient_x = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            gradient_y = self.fn_grad2(x_value-gamma*x_step, y_value-gamma*y_step)
            #if the gradient is not zero, the x and y values are adjusted by:
            # 1. The step_size(eta)*respective gradient at a 'future' position
            # 2. An additional adjustment = last_step size*alpha(alpha set at 0.9)
            x_step = alpha * x_step + eta * gradient_x
            x_value += -x_step
            x_path.append(x_value)
            y_step = alpha * y_step + eta * gradient_y
            y_value += -y_step
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)
        #Output when running the algorthm
        if np.isnan(gradient_x) or np.isnan(gradient_y):
            print('Diverged at step size= {} after {} steps'.format(np.round(eta, 4), num_iters))
        elif np.abs(gradient_x) > tol or np.abs(gradient_y) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps, step size= {}. Loss fn= {} at (x, y) = {}'
                  .format(num_iters, np.round(eta, 4), np.round(loss_this, 4),
                          (np.round(x_value, 4), np.round(y_value, 4))))
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        self.loss_fn_min = loss_this
        self.x_at_min = x_value
        self.y_at_min = y_value
        self.n_iter = n_iter
        self.num_iters = num_iters
        #self.x_path = np.array(x_path)

# %% Setting up the plotting algorithm for plain vanilla gradient descent
    def find_min(self, x_init, y_init, max_iter, eta, tol):
        """The algorithm is the plain vanilla but with a plot function"""
        # Initialise lists to store results
        x_path = []
        y_path = []
        loss_path = []
        # Initializing the original values of x, y, f(x,y) and n_iter
        x_value = x_init
        y_value = y_init
        num_iters = 0
        # Storing these initial values to the lists
        x_path.append(x_value)
        y_path.append(y_value)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        # Performing the loop for determinimg the minimum of the function
        for i in range(max_iter):
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                #Loops till the partial derivatives are zero or function diverges
                break
            num_iters = num_iters + 1
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            x_value += -eta * gradient_x
            x_path.append(x_value)
            y_value += -eta * gradient_y
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)

        loss_min = "{:.2f}".format(loss_this)
        iterations = "{}".format(num_iters)
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        fig2 = plt.figure(figsize=(10, 10))
        ax1 = plt.axes(projection='3d')
        ax1.set_xlabel('x')#, labelpad=15)
        ax1.set_ylabel('y')#, labelpad=15)
        ax1.set_zlabel('loss function')#, labelpad=15)
        ax1.text(x_value, y_value, loss_this, (loss_min, iterations), color='b', fontsize=15, style='italic')
        ax1.set_title('Interactive six hump graph', loc='left')
        ax1.scatter3D(self.x_path, self.y_path, self.loss_path, c=self.loss_path, cmap='hsv')

# %% Setting up the plotting algorithm for momentum gradient descent
    def find_min2(self, x_init, y_init, max_iter, eta, tol, alpha):
        """The algorithm is the plain vanilla but with a plot function"""
        # Initialise lists to store results
        x_path = []
        y_path = []
        loss_path = []
        # Initializing the original values of x, y, f(x,y) and n_iter
        x_value = x_init
        y_value = y_init
        num_iters = 0
        # Storing these initial values to the lists
        x_path.append(x_value)
        y_path.append(y_value)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        # Additional parameters. This refer to the last step taken by x_value and y_value
        x_step = 0
        y_step = 0
        for i in range(max_iter):
            gradient_y = self.fn_grad2(x_value, y_value)
            gradient_x = self.fn_grad1(x_value, y_value)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                #Loops till the partial derivatives are either zero or function diverges
                break
            num_iters = num_iters + 1
            gradient_x = self.fn_grad1(x_value, y_value)
            gradient_y = self.fn_grad2(x_value, y_value)
            #if the gradient is not zero, the x and y values are adjusted by:
            # 1. The step_size(eta)*respective gradient
            # 2. An additional adjustment = last_step size*alpha(alpha set at 0.9)
            x_step = alpha * x_step + eta * gradient_x
            x_value += -x_step
            x_path.append(x_value)
            y_step = alpha * y_step + eta * gradient_y
            y_value += -y_step
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)

        loss_min = "{:.2f}".format(loss_this)
        iterations = "{}".format(num_iters)
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        fig2 = plt.figure(figsize=(10, 10))
        ax1 = plt.axes(projection='3d')
        ax1.set_xlabel('x')#, labelpad=15)
        ax1.set_ylabel('y')#, labelpad=15)
        ax1.set_zlabel('loss function')#, labelpad=15)
        ax1.text(x_value, y_value, loss_this, (loss_min, iterations), color='b', fontsize=15, style='italic')
        ax1.set_title('Interactive six hump graph', loc='left')
        ax1.scatter3D(self.x_path, self.y_path, self.loss_path, c=self.loss_path, cmap='hsv')

# %% Setting up the plotting algorithm for Nestrov's Accelerated gradient descent
    def find_min3(self, x_init, y_init, max_iter, eta, tol, alpha):
        """The algorithm is the plain vanilla but with a plot function"""
        # Initialise lists to store results
        x_path = []
        y_path = []
        loss_path = []
        # Initializing the original values of x, y, f(x,y) and n_iter
        x_value = x_init
        y_value = y_init
        num_iters = 0
        # Storing these initial values to the lists
        x_path.append(x_value)
        y_path.append(y_value)
        loss_this = self.fn_loss(x_value, y_value)
        loss_path.append(loss_this)
        x_step = 0
        y_step = 0
        for i in range(max_iter):
            # i starts from 0 so add 1
            gamma = 1 - 3 / (i + 1 + 5)
            #Gamma represents the weight given to the last step size when
            #determinimg the approximate future position.
            #The gradient is calculated at the approximate future x and y.
            gradient_x = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            gradient_y = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            if ((np.abs(gradient_x) < tol and np.abs(gradient_y) < tol) or
                    np.isnan(gradient_x) or np.isnan(gradient_y)):
                break
            num_iters = num_iters + 1
            gradient_x = self.fn_grad1(x_value-gamma*x_step, y_value-gamma*y_step)
            gradient_y = self.fn_grad2(x_value-gamma*x_step, y_value-gamma*y_step)
            #if the gradient is not zero, the x and y values are adjusted by:
            # 1. The step_size(eta)*respective gradient at a 'future' position
            # 2. An additional adjustment = last_step size*alpha(alpha set at 0.9)
            x_step = alpha * x_step + eta * gradient_x
            x_value += -x_step
            x_path.append(x_value)
            y_step = alpha * y_step + eta * gradient_y
            y_value += -y_step
            y_path.append(y_value)
            loss_this = self.fn_loss(x_value, y_value)
            loss_path.append(loss_this)

        loss_min = "{:.2f}".format(loss_this)
        iterations = "{}".format(num_iters)
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        fig2 = plt.figure(figsize=(10, 10))
        ax1 = plt.axes(projection='3d')
        ax1.set_xlabel('x')#, labelpad=15)
        ax1.set_ylabel('y')#, labelpad=15)
        ax1.set_zlabel('loss function')#, labelpad=15)
        ax1.text(x_value, y_value, loss_this, (loss_min, iterations), color='b', fontsize=15, style='italic')
        ax1.set_title('Interactive six hump graph', loc='left')
        ax1.scatter3D(self.x_path, self.y_path, self.loss_path, c=self.loss_path, cmap='hsv')
