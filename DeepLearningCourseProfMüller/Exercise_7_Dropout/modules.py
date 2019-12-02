import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        self.p = p
        
    def forward(self, x):
        # In training mode, set each value 
        # independently to 0 with probability p
        # In evaluation mode, return the
        # unmodified input
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return x * binomial.sample(x.size()) * (1.0/(1-self.p))          #return modified during
                                                                             #training
        return x                                                             #return unmodified when not
                                                                             #training    
            
             
        
    