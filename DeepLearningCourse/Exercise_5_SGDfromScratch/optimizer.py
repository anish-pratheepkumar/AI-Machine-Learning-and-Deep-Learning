class SGD:
    
    def __init__(self, params, lr=1e-3):
        #super().__init__()
        # TODO: save the input argument 'params' as list
        #       and store the learning rate
        self.params = list(params)                            #params is a generator, creating list from                                                               #params so that we can access data 
        self.lr = lr
            
    def step(self):
        # TODO: update the parameters' gradients according to
        #       gradient descent with step size lr;
        #       for modifying the parameters use the function
        #       'copy_' on the parameters' 'data' attributes
        for param in self.params:
            update = self.lr*param.grad.data               #gradient descent update
            param.data.copy_(param.data-update)          #using copy fn to update the weigts 
            
    def zero_grad(self):
        # TODO: set all the parameters' gradients to zero
        #       by calling 'zero_' on the parameters' gradients
        for param in self.params:
            param.grad.zero_()
            #param.grad.data = torch.zeros_like(param.grad.data)
        