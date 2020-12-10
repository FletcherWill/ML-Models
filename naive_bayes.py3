import pandas as pd
import numpy as np
from collections import defaultdict

class NaiveBayes:

  def __init__(self, categorical_features):
    self.categorical_features = categorical_features #Categorical features the model will consider
    self.class_probabilities = None #self.class_probabilities[i] = P(C = i)
    self.probabilities = None #self.probabilities[i][F][j] = P(F = j | C = i)


  def train(self, df, target, additive_smoothing_value = 0):
    
    prob_not_vulnerable = (df[target].sum()+additive_smoothing_value)/(df.shape[0]+ 2*additive_smoothing_value)
    self.class_probabilities = {0: 1 - prob_not_vulnerable, 1: prob_not_vulnerable}

    num_ones = df[target].sum()
    num_zeros = df.shape[0] - num_ones
    self.probabilities = {0: {}, 1: {}}
    #cycle through the features and calculate probabilities
    for feature in self.categorical_features:
      counts_zero = defaultdict(lambda: additive_smoothing_value / (df.shape[0] + additive_smoothing_value * 2) + 10**(-10)) #Default dicts so that if a value is not seen in training
      counts_one = defaultdict(lambda: additive_smoothing_value / (df.shape[0] + additive_smoothing_value * 2) + 10**(-10))  #it's probability is determined by the smoothing value plus a small epsilon so that if the smoothing value is zero, we don't take the log of zero
      for i in range(df.shape[0]):
        if df.iloc[i][target] == 0:
          feature_val = df.iloc[i][feature]
          if feature_val in counts_zero:
            counts_zero[feature_val] += 1
          else:
            counts_zero[feature_val] = 1
        else:
          feature_val = df.iloc[i][feature]
          if feature_val in counts_one:
            counts_one[feature_val] += 1
          else:
            counts_one[feature_val] = 1
      for val in counts_zero:
        counts_zero[val] = (counts_zero[val] + additive_smoothing_value) / (num_zeros + (additive_smoothing_value * len(counts_zero)))
      for val in counts_one:
        counts_one[val] = (counts_one[val] + additive_smoothing_value) / (num_ones + (additive_smoothing_value * len(counts_one)))
      self.probabilities[0][feature] = counts_zero
      self.probabilities[1][feature] = counts_one


  def get_probability(self, feature_name, feature_value, clas):
    return self.probabilities[clas][feature_name][feature_value]

  def get_class_probability(self, clas):
    return self.class_probabilities[clas]

  def test(self, datapoint):
    prob_0 = np.log(self.get_class_probability(0))
    for feature in self.categorical_features:
      prob_0 += np.log(self.get_probability(feature, datapoint[feature], 0))
    prob_1 = np.log(self.get_class_probability(1))
    for feature in self.categorical_features:
      prob_1 += np.log(self.get_probability(feature, datapoint[feature], 1))
    if prob_0 > prob_1:
      return 0
    else:
      return 1
