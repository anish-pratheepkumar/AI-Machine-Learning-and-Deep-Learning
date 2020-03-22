import torch.nn as nn

##################################################
# Implement your network architecture in CIFARNet
##################################################

class CIFARNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(3, 6, 5), nn.BatchNorm2d(6),nn.ReLU(),                                          nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv_layer2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.BatchNorm2d(16), nn.ReLU(),                                        nn.MaxPool2d(2,2), nn.Dropout2d(p=0.05))
               
        self.lin_layer1 = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(16*5*5, 120), nn.ReLU())
        self.lin_layer2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.lin_layer3 = nn.Sequential(nn.Linear(84, 10), nn.ReLU())
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.reshape(-1, 5*5*16)
        x = self.lin_layer1(x)
        x = self.lin_layer2(x)
        x = self.lin_layer3(x)
        return x
        pass

       