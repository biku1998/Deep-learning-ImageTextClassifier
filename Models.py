import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt


class SigmoidNeuron:


    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self,x):
        return np.dot(x,self.w.T) + self.b

    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))

    def grad_w_ce(self,x,y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y)*x

    def grad_b_ce(self,x,y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y)

    def grad_w_mse(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b_mse(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)

    def fit(self,X,Y,epochs = 1, learning_rate = 1,initialize = True,loss_func = "mse",display_loss = False):
        # initliaze w, b
        if initialize:
            self.w = np.random.randn(1,X.shape[1])
            self.b = 0.1 # just randomly picked.

        if display_loss:
            loss = {} # to store all the loss value for plotting.

        for i in tqdm(range(epochs),total = epochs , unit = 'epoch'):

            dw = 0
            db = 0

            for x,y in zip(X,Y):

                if loss_func == "mse":
                    dw += self.grad_w_mse(x,y)
                    db += self.grad_b_mse(x,y)
                elif loss_func == "ce":
                    dw += self.grad_w_ce(x,y)
                    db += self.grad_b_ce(x,y)

            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                if loss_func == "mse":
                    loss[i] = mean_squared_error(Y,Y_pred)
                elif loss_func == "ce":
                    loss[i] = log_loss(Y,Y_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            if loss_func == "mse":
                plt.ylabel("mean_squared_error")
            elif loss_func == "ce":
                plt.ylabel("log_loss")
            plt.grid()
            plt.show()

    def predict(self,X):
        Y_pred = []
        for x in X:
            res = self.sigmoid(self.perceptron(x))
            Y_pred.append(res)
        return np.array(Y_pred)




            
            
            