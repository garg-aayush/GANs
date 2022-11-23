import torch
import torch.nn as nn
from typing import List
from torchsummary import summary

class CNNBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: int=2 ) -> nn.Module:
        
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


# takes 4 blocks
# Discriminator has 4 CNN layers
# 256X256 --> 26X26
class Discriminator(nn.Module):
    def __init__(self,
                in_channels: int = 3,
                features: List[int] = [64, 128, 256, 512]): 
        
        super().__init__()

        # takes both input and target and concatenates along the channels
        self.initial = nn.Sequential(
                        nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                        nn.LeakyReLU(0.2)

        )

        # make other 3 layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        # make it single channel
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.initial(x)
        return self.model(x)

 
# test case
def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    out = model(x,y)
    #print(model)
    summary(model, [(3,256,256), (3,256,256)])
    
    print('out shape: ', out.shape)


if __name__ == '__main__':
    test()
 