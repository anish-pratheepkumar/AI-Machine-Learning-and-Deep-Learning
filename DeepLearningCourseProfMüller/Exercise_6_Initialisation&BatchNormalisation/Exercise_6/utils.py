import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        y = m.weight.data.size(1)              #10 nos i/p size for 1st layer     
        z = m.weight.data.size(0)              #100 nos o/p size for first layer
        std = torch.sqrt(torch.tensor(2.)/y)
        return(torch.randn(y,z)*std)
        pass
        

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is normalized and zero centered
        theta_mu = torch.zeros(num_channels)     #we need 10 values, each batch(4x10) should be reduced to represent as a single element i.e, reduce 4x10 to 10
        theta_sigma = torch.ones(num_channels)
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            # specify behavior at training time
            
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = torch.mean(x,0)
                self.running_var = torch.var(x,0)
                x_n = (x-self.running_mean)*(1.0/torch.sqrt(self.running_var+self.eps))
               
                return x_n
                pass
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                #self.running_mean = self.running_mean*0.9 + torch.mean(x,0)*0.1
                #self.running_var = self.running_var*0.9 + torch.var(x,0)*0.1
                self.running_mean = (0.9*self.running_mean+0.1*torch.mean(x,0))
                self.running_var = (0.9*self.running_var+0.1*torch.var(x,0))
                x_n = (x-torch.mean(x,0))*(1.0/torch.sqrt(torch.var(x,0)+self.eps))
                x = x_n*self.theta_sigma + self.theta_mu
                return x
                pass
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                x_n = (x-torch.mean(x))*(1.0/torch.sqrt(torch.var(x)+self.eps))
                x = x_n*self.theta_sigma + self.theta_mu
                return x
                pass
            else:
                # use running stats for normalization
                x_n = (x-self.running_mean)*(1.0/torch.sqrt(self.running_var+self.eps))
                x = x_n*self.theta_sigma + (self.theta_mu)
                return x
                pass
    