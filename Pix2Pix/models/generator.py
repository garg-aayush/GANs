import torch
import torch.nn as nn
from torchsummary import summary

class CNNBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                down: bool=True,
                act: str="relu",
                use_dropout: bool=False) -> nn.Module:

        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 4, 2, padding=1, bias=False, padding_mode="reflect")
                        if down else
                        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU() if act=="relu" else nn.LeakyReLU(0,2)
        )
        
        self.use_dropout = use_dropout
        self.droput = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.droput(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self,
                in_channels: int=3,
                features: int=64) -> nn.Module:
        super().__init__()
        self.initial_down = nn.Sequential(
                            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
                            nn.LeakyReLU(0.2)
        )

        self.down1 = CNNBlock(features, features*2, down=True, act="leaky", use_dropout=False)      #64X64
        self.down2 = CNNBlock(features*2, features*4, down=True, act="leaky", use_dropout=False)    #32X32
        self.down3 = CNNBlock(features*4, features*8, down=True, act="leaky", use_dropout=False)    #16X16
        self.down4 = CNNBlock(features*8, features*8, down=True, act="leaky", use_dropout=False)    #8X8
        self.down5 = CNNBlock(features*8, features*8, down=True, act="leaky", use_dropout=False)    #4X4
        self.down6 = CNNBlock(features*8, features*8, down=True, act="leaky", use_dropout=False)    #2X2

        self.bottleneck = nn.Sequential(
                            nn.Conv2d(features*8, features*8,  4, 2, 1, padding_mode="reflect"), 
                            nn.ReLU() #1X1
        )

        self.up0 = CNNBlock(features*8, features*8, down=False, act="relu", use_dropout=True)
        self.up1 = CNNBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up2 = CNNBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = CNNBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False)
        self.up4 = CNNBlock(features*8*2, features*4, down=False, act="relu", use_dropout=False)
        self.up5 = CNNBlock(features*4*2, features*2, down=False, act="relu", use_dropout=False)
        self.up6 = CNNBlock(features*2*2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
                            nn.ConvTranspose2d(features*2, in_channels, 4,2,1),
                            nn.Tanh(),
        )
                            

    def forward(self, x):
        d0 = self.initial_down(x)
        d1 = self.down1(d1)
        d2 = self.down2(d2)
        d3 = self.down3(d3)
        d4 = self.down4(d4)
        d5 = self.down5(d5)
        d6 = self.down6(d6)
        botteneck = self.bottleneck(d6)
        up0 = self.up0(botteneck)
        up1 = self.up2(torch.cat([up1, d6], dim=1))
        up2 = self.up3(torch.cat([up1, d5], dim=1))
        up3 = self.up4(torch.cat([up2, d4], dim=1))
        up4 = self.up5(torch.cat([up3, d3], dim=1))
        up5 = self.up6(torch.cat([up4, d2], dim=1))
        up6 = self.up7(torch.cat([up5, d1], dim=1))
        return self.final_up(torch.cat([up6, d1], dim=1))


# test case
def test():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features=64)
    out = model(x)
    summary(model, (3,256,256))
    print('out shape: ', out.shape)


if __name__ == '__main__':
    test()
         

