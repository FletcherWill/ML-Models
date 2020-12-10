'''
SVM class which initializes with a kernel type. 
SVMs have a get_kernel function for creating the kernel, a training function, and a predict function.
'''
class SVM:

  '''
  Initializes the kernel for the SVM as well as variables for data, labels, alphas, and bias.
  '''
  def __init__(self, kernel_type = "Polynomial", kernel_hyperparams = 1):
    self.kernel = self.get_kernel(kernel_type, kernel_hyperparams)
    self.data = None
    self.labels = None
    self.alphas = None
    self.bias = None

  '''
  returns the kernel function based on the kernel type passed into init.
  '''
  def get_kernel(self, type, hyperparams):
    if type == "Polynomial":
      degree = hyperparams
      def ker(v,w):
        dot = 0
        for i in range(len(v)):
          dot += v[i]*w[i]
        return (1 + dot) ** degree
    if type == "Radial":
      gamma = hyperparams
      def ker(v,w):
        sum = 0
        for i in range(len(v)):
          sum += (v[i]-w[i]) ** 2
        return math.exp(-gamma*sum)
    return ker

  '''
  calculates classifier function of an input x, without classifying.
  '''
  def f(self, x):
    sum = self.bias
    for i in range(len(self.alphas)):
      sum += self.labels[i]*self.alphas[i]*self.kernel(x, self.data[i])
    return sum

  '''
  trains the SVM given a dataset, labels for the dataset, and the hyperparameters:
  C, tol, and max_passes.
  This algorithm is the simplified SMO algorithm from http://cs229.stanford.edu/materials/smo.pdf
  '''
  def train(self, data, labels, C, tol, max_passes):

    '''
    Gets a random index j not equal to i
    '''
    def get_j(i, leng):
      j = i
      while j == i:
        j = random.randrange(leng)
      return j

    '''
    gets mu value of 2 datapoints.
    '''
    def get_mu(i, j):
      return 2*self.kernel(self.data[i],self.data[j]) - self.kernel(self.data[i],self.data[j]) - self.kernel(self.data[i],self.data[i]) - self.kernel(self.data[j],self.data[j])

    '''
    calculates new alpha_j
    '''
    def get_new_aj(j, L, H, old_aj, mu, error_i, error_j):
      aj = old_aj - self.labels[j]*(error_i-error_j)/mu
      if aj > H:
        return H
      elif aj < L:
        return L
      else:
        return aj

    '''
    calculates new bias value
    '''
    def get_b(b_1, b_2, new_ai, new_aj):
      if 0 < new_ai and new_ai < C:
        return b_1
      elif 0 < new_aj and new_aj < C:
        return b_2
      else:
        return (b_1+b_2)/2

    self.labels = labels
    self.alphas = [0] * data.shape[0]
    self.data = data
    self.bias = 0
    passes = 0
    while passes < max_passes:
      num_changed_alphas = 0
      for i in range(len(self.alphas)):
        error_i = self.f(data[i]) - labels[i]
        if (labels[i]*error_i < -tol and self.alphas[i] < C) or (labels[i]*error_i > tol and self.alphas[i] > 0):
          j = get_j(i, len(data))
          error_j = self.f(data[j]) - labels[j]
          old_ai, old_aj = self.alphas[i], self.alphas[j]
          if labels[i] == labels[j]:
            L = max(0, old_ai + old_aj - C)
            H = min(C, old_ai + old_aj)
          else:
            L = max(0, old_aj - old_ai)
            H = min(C, C + old_aj - old_ai)
          mu = get_mu(i,j)
          if L != H and mu < 0:
            new_aj = get_new_aj(j, L, H, old_aj, mu, error_i, error_j)
            if abs(new_aj-old_aj) >= 10^(-5):
              new_ai = old_ai+self.labels[i]*self.labels[j]*(old_aj-new_aj)
              b_1 = self.bias - error_i - self.labels[i]*(new_ai - old_ai)*self.kernel(data[i],data[i]) - self.labels[j]*(new_aj - old_aj)*self.kernel(data[i],data[j])
              b_2 = self.bias - error_j - self.labels[i]*(new_ai - old_ai)*self.kernel(data[i],data[j]) - self.labels[j]*(new_aj - old_aj)*self.kernel(data[j],data[j])
              self.bias = get_b(b_1, b_2, new_ai, new_aj)
              self.alphas[i] = new_ai
              self.alphas[j] = new_aj
              num_changed_alphas += 1
      #print(num_changed_alphas)
      if num_changed_alphas == 0:
        passes += 1
      else:
        passes = 0
    self.bias = self.bias

  '''
  predicts a single datapoint
  '''
  def predict_helper(self, datapoint):
    return 2*int(self.f(datapoint) >= 0) - 1

  '''
  predicts all datapoints of passed in dataset
  '''
  def predict(self, data):
    predictions = []
    for dp in data:
      predictions.append(self.predict_helper(dp))
    return predictions
