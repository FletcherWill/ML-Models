import pandas as pd
import numpy as np
import math

class LogisticModel:

  def __init__(self):
    self.weights = False
    self.isTrained = False

  def train(self, X, y, step_size, num_iterations):
    self.isTrained = True
    self.weights = np.zeros(X.shape[1]+1)
    N = X.shape[0]
    
    average_loss_list = []

    for ni in range(num_iterations):
      
      total_difference = 0
      for i in range(N):
        data_point = X[i]
        y_hat = self.weights[-1]
        for j in range(len(data_point)):
          y_hat += self.weights[j] * data_point[j]
        y_hat = 1/(1+math.exp(-y_hat))
        total_difference += y[i] - y_hat
      self.weights[-1] += (step_size / N * total_difference)
      
      for k in range(X.shape[1]):
        total_difference = 0
        for i in range(N):
          data_point = X[i]
          y_hat = self.weights[-1]
          for j in range(len(data_point)):
            y_hat += self.weights[j] * data_point[j]
          y_hat = 1/(1+math.exp(-y_hat))
          total_difference += ((y[i] - y_hat) * data_point[k])
        self.weights[k] += (step_size / N * total_difference)

      average_loss_list.append(self.get_average_loss(X,y))
    
    return average_loss_list


  def predict(self, X):
    if not self.isTrained:
      return None
    N = len(X)
    output = np.zeros(N)
    for i in range(N):
      data_point = X[i]
      y_hat = self.weights[-1]
      for j in range(len(data_point)):
        y_hat += self.weights[j] * data_point[j]
      y_hat = 1/(1+math.exp(-y_hat))
      if y_hat >= 0.5:
        output[i] = 1
      else:
        output[i] = 0
    return np.asarray(output)


  def get_weights(self):
    return self.weights

  def get_average_loss(self, X, y):
    if not self.isTrained:
      return None
    N = X.shape[0]
    total_loss = 0
    for i in range(N):
      data_point = X[i]
      y_hat = self.weights[-1]
      for j in range(len(data_point)):
        y_hat += self.weights[j] * data_point[j]
      y_hat = 1/(1+math.exp(-y_hat))
      total_loss += -(y[i]*np.log(y_hat)+(1-y[i])*np.log(1-y_hat))
    return total_loss / N
