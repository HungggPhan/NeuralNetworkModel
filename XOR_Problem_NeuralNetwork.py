import numpy as np

#build activation functions and derivative of per activation function

#sigmoid
def sigmoid(x):
  return 1/(1+np.exp(-x))
#derivative of sigmoid function
def sigmoid_derivative(x):
  return x*(1-x)

#tanh function
def tanh(x):
  return np.tanh(x)
#derivative of tanh function
def tanh_derivative(x):
  return 1-x**2

#ReLU function:
def ReLU(x):
  return max(0,x)

#derivative of ReLU function with x>0 always is 1

class NeuralNetwork:
  def __init__(self,layers,theta):
    #layers[] is an array, per value is number of nodes on that layer
    #theta is learning rate
    self.layers=layers
    self.theta=theta

    #define Weights and bias is array
    self.W=[]
    self.b=[]

    # Create the first value for layers on model 
    for i in range(0, len(layers)-1):
      w_ = np.random.randn(layers[i], layers[i+1])
      b_ = np.zeros((layers[i+1], 1))
      self.W.append(w_/layers[i])
      self.b.append(b_)

  # Train model per eposch
  def fit_partial(self, x, y):
    A = [x]
    # feedforward
    out = A[-1]
    # keep the input layer values, caculate values of hidden layers and output
    for i in range(0, len(self.layers) - 1):
      out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
      A.append(out)

    # backpropagation
    y = y.reshape(-1, 1)
    dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
    dW = []
    db = []
    #caculate derivative of per node on layers
    #this model I use sigmoid function, the others are same.
    for i in reversed(range(0, len(self.layers)-1)):
      dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
      db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
      dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
      dW.append(dw_)
      db.append(db_)
      dA.append(dA_)
    # Reverse dW, db
    dW = dW[::-1]
    db = db[::-1]
    # Gradient descent : update values of Weights, bias
    for i in range(0, len(self.layers)-1):
      self.W[i] = self.W[i] - self.theta * dW[i]
      self.b[i] = self.b[i] - self.theta * db[i]

  #predict y: output value
  def predict(self, X):
    for i in range(0, len(self.layers) - 1):
      X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
    return X
  # Loss caculate
  def calculate_loss(self, X, y):
    y_predict = self.predict(X)
    return np.sum((y_predict-y)**2)/2
  
  #fit model by loop action train the data epochs time
  def fit(self, X, y, epochs, verbose=10):
    for epoch in range(0, epochs):
      self.fit_partial(X, y)
      loss = self.calculate_loss(X, y)
      if(epoch%verbose==1000):
        print("Epoch {}, loss {}".format(epoch, loss))

#create data to train model
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

#create model : 2 nodes on input layer,
#1 hidden layer with 2 nodes, output layer.
#choose learning rate = about 0.05 to 1
ANN=NeuralNetwork([2,2,1],theta=1)
ANN.fit(X,y,epochs=100000,verbose=5000)
print("Predict with data: ",ANN.predict([1,0]))