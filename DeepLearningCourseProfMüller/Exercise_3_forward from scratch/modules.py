import numpy as np


class Module:
    
    def forward(self, *args, **kwargs):
        pass

    
class Network(Module):
    
    def __init__(self, layers=[]):
        # store the layers passed in the constructor in your Network object
        self.layers = layers
        pass
    
    def forward(self, x):
        # for executing the forward pass, run the forward passes of each
        # single layer and pass the output as input to the next layer
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x
        pass
    
    
    def add_layer(self, layer):
        # append layer at the end of the already existing layer
        self.layers.append(layer)
        pass

    
class LinearLayer(Module):
    
    def __init__(self, W, b):
        # store parameters W and b
        self.weights = W
        self.bias = b
        pass
    
    def forward(self, x):
        # compute the linear transformation x -> Wx + b
        return np.dot(self.weights,x) + self.bias
        pass

    
class Sigmoid(Module):
    
    def forward(self, x):
        # implement the sigmoid
        return np.exp(x) / (1 + np.exp(x))
        pass

    
class ReLU(Module):
    
    def forward(self, x):
        # implement a ReLU
        return np.maximum(x,0)
        pass

    
class Loss(Module):
    
    def forward(self, prediction, target):
        pass


class MSE(Loss):
    
    def forward(self, prediction, target):
        # implement MSE loss
        return np.square(prediction - target).mean()
        pass


class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        # implement cross entropy loss
        softmax = np.exp(prediction)/np.sum(np.exp(prediction))
        return -np.log(softmax)
        pass
