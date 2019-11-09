import numpy as np
import toolbox as tb
from toolbox import Tensor


class Module:
    
    def forward(self, *args, **kwargs):
        pass
    
    def zero_grad(self):
        pass
    
    def parameters(self):
        return []


class Network(Module):
    
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers
    
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()
    
    def parameters(self):
        params = []
        for l in self.layers:
            params = params + l.parameters()
        return params


class LinearLayer(Module):
    
    def __init__(self, W, b):
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)
    
    def forward(self, x):
        return self.W @ x + self.b
    
    def zero_grad(self):
        self.W.zero_grad()
        self.b.zero_grad()
    
    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    
    def forward(self, x):
        return tb.relu(x)


class Loss(Module):
    
    def forward(self, prediction, target):
        pass


class MSE(Loss):
    
    def forward(self, prediction, target):
        return ((prediction + Tensor(-1.) * target) ** 2).mean()


class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        return Tensor(-1.) * tb.log(tb.exp(prediction[target])/tb.exp(prediction).sum())

