import numpy as np


class Module:
    
    def forward(self, *args, **kwargs):
        pass

    
class Network(Module):
    
    def __init__(self, layers=[]):
        # store the layers passed in the constructor in your Network object
        self.layers = layers
            
    def forward(self, x):
        # for executing the forward pass, run the forward passes of each
        # single layer and pass the output as input to the next layer
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x
            
    def add_layer(self, layer):
        # append layer at the end of the already existing layer
        self.layers.append(layer)
        pass

    
class LinearLayer(Module):
    
    def __init__(self, W, b):
        # store parameters W and b
        self.weights = W
        self.bias = b
            
    def forward(self, x):
        # compute the linear transformation x -> Wx + b
        return self.weights @ x + self.bias
       
    
class Sigmoid(Module):
    
    def forward(self, x):
        # implement the sigmoid
        return np.exp(x) / (1 + np.exp(x))
        
    
class ReLU(Module):
    
    def forward(self, x):
        # implement a ReLU
        return np.maximum(x,0)
        
        
class Loss(Module):
    
    def forward(self, prediction, target):
        pass

class MSE(Loss):
    
    def forward(self, prediction, target):
        # implement MSE loss
        return np.square(prediction - target).mean()
        
class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        # implement cross entropy loss
        prediction = prediction - prediction.max()
        softmax = np.exp(prediction[target])/np.sum(np.exp(prediction))
        return -np.log(softmax)
       