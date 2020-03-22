import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        y = m.weight.data.shape[0]       #100 nos o/p size for first layer           
        z = m.weight.data.shape[1]       #10 nos i/p size for 1st layer
        std = torch.sqrt(torch.tensor(2.)/z)
        m.weight = nn.Parameter(torch.randn(y,z)*std)
        m.bias = nn.Parameter(torch.zeros(y))
               

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is normalized and zero centered
        theta_mu = torch.zeros(num_channels)     #we need 10 values, each batch(4x10) should be reduced                                                     to represent as a single element i.e, reduce 4x10 to 10
        theta_sigma = torch.ones(num_channels)
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            # specify behavior at training time
            mean = x.mean(dim=0)
            var = x.var(dim=0)            
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = nn.Parameter(mean, requires_grad=False)
                self.running_var = nn.Parameter(var, requires_grad=False)               
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                #self.running_mean = self.running_mean*0.9 + torch.mean(x,0)*0.1
                #self.running_var = self.running_var*0.9 + torch.var(x,0)*0.1
                self.running_mean.data = 0.1 * mean + 0.9 * self.running_mean
                self.running_var.data = 0.1 * var + 0.9 * self.running_var
        else:   #during inference
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                mean = x.mean(dim=0)
                var = x.var(dim=0)
            else:
                # use running stats for normalization
                mean = self.running_mean
                var = self.running_var
       
        x_norm = (x - mean)/(var + self.eps).sqrt()
        return x_norm * self.theta_sigma + self.theta_mu    
        
    