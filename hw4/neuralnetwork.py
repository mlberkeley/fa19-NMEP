import numpy as np

class NeuralNet:

    #constructor, we will hardcode this to a 1 hidden layer network, for simplicity
    #the problem we will grade on is differentiating 0 and 1s
    #Some things/structuure may need to be changed. What needs to stay consistant is us being able to call
    #forward with 2 arguments: a data point and a label. Strange architecture, but should be good for learning
    def __init__(self, input_size=784, hidden_size=100, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #YOUR CODE HERE, initialize appropriately sized weights/biases with random paramters
        self.weight1 = 
        self.bias1 = 
        self.weight2 = 
        self.bias2 = 

    #Potentially helpful, np.dot(a, b), also @ is the matrix product in numpy (a @ b)

    #loss function, implement L1 loss
    #YOUR CODE HERE
    def loss(self, y0, y1):

    #relu and sigmoid, nonlinear activations
    #YOUR CODE HERE
    def relu(self, x):

    #You also may want the derivative of Relu and sigmoid

    def sigmoid(self, x):

    #forward function, you may assume x is correct input size
    #have the activation from the input to hidden layer be relu, and from hidden to output be sigmoid
    #have your forward function call backprop: we won't be doing batch training, so for EVERY SINGLE input,
    #we will update our weights. This is not always (maybe not even here) possible or practical, why?
    #Also, normally forward doesn't take in labels. Since we'll have forward call backprop, it'll take in labels
    #YOUR CODE HERE
    def forward(self, x, label):

    #implement backprop, might help to have a helper function update weights
    #Recommend you check out the youtube channel 3Blue1Brown and their video on backprop
    #YOUR CODE HERE
    def backprop(self, x, label): #What else might we need to take in as arguments? Modify as necessary

        #Compute the gradients first
        #First will have to do with combining derivative of sigmoid, output layer, and what else?
        #np.sum(x, axis, keepdims) may be useful

        #Update your weights and biases. Use a learning rate of 0.1, and update on every call to backprop
        lr = .1

    
