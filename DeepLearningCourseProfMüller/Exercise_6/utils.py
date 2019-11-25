import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        pass
        

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set mu and sigma such that the output of forward
        # initially is normalized and zero centered
        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            # specify behavior at training time
            if self.running_mean is None:
                # set the running stats to stats of x
                pass
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                pass
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                pass
            else:
                # use running stats for normalization
                pass
    