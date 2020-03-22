import torch
import torch.nn.functional as F


class Conv2d():
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        
    def forward(self, x):
        # Implement the cross-corelation of x and self.kernel
        # using self.padding and self.stride
        h = ((x.shape[1]-self.kernel.shape[2]+2*self.padding)*(1/self.stride))+1
        w = ((x.shape[2]-self.kernel.shape[3]+2*self.padding)*(1/self.stride))+1
        output = torch.zeros(self.kernel.shape[0],int(h),int(w))
        
        padded_x = torch.zeros(x.shape[0],x.shape[1]+2*self.padding,x.shape[2]+2*self.padding)
        padded_x[:,self.padding:-self.padding,self.padding:-self.padding] = x

        
        for i in range(self.kernel.shape[0]):
            for j in range(int(h)):
                for k in range(int(w)):
                    start_row_indx = 0 + (j*self.stride) 
                    end_row_indx = (self.kernel.shape[2]) + (j*self.stride)
                    start_col_indx = 0 + (k*self.stride)
                    end_col_indx = (self.kernel.shape[3]) + (k*self.stride)
                    
                    patch = padded_x[:,start_row_indx:end_row_indx,
                                         start_col_indx:end_col_indx]
                  
                    conv_mul = patch * self.kernel[i]
                    conv_sum = torch.sum(conv_mul)
                    output[i,j,k] = conv_sum

        return output        

    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

      
        
        
        
        
        
        
        
        
        
        
        
        
        