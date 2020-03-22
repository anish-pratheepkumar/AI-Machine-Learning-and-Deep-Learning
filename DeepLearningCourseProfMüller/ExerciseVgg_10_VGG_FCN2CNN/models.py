import torch.nn as nn
from torchvision.models import vgg16 as VGG


class ConvVGG(nn.Module):
    
    def __init__(self):
        super().__init__()
        vgg = VGG(pretrained=True)
        self.features = vgg.features
        
        ##############################################
        # Replace the code such that self.classifier
        # behaves in the same way as vgg.classifier
        # with the only difference being that the
        # linear layers are replaced by appropriate
        # convolutional layers

        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, 7),vgg.classifier[1],vgg.classifier[2],
                                        nn.Conv2d(4096, 4096, 1),vgg.classifier[4],vgg.classifier[5],
                                        nn.Conv2d(4096, 1000, 1))
        
        self.classifier[0].weight.data =                                                    vgg.classifier[0].weight.data.view(self.classifier[0].weight.size())
        self.classifier[0].bias.data = vgg.classifier[0].bias.data.view(self.classifier[0].bias.size())
        
        self.classifier[3].weight.data = vgg.classifier[3].weight.data.view(self.classifier[3].weight.size())    
        self.classifier[3].bias.data = vgg.classifier[3].bias.data.view(self.classifier[3].bias.size())
        
        self.classifier[6].weight.data = vgg.classifier[6].weight.data.view(self.classifier[6].weight.size())
        self.classifier[6].bias.data = vgg.classifier[6].bias.data.view(self.classifier[6].bias.size())
        
        
        ##############################################
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.sum((2, 3))
