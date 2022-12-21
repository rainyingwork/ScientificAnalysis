import torch.nn as nn
import torch.nn.functional as functional

# class Reshape(nn.Module):
#
#     def __init__(self, *args):
#         super(Reshape, self).__init__()
#         self.shape = args
#
#     def forward(self, x):
#         return x.view(self.shape)

class ResidualBlock(nn.Module):
    
    def __init__(self, inChannel, outChannel, stride=1) :
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
        )
        self.shortcut = nn.Sequential()
        
        if stride != 1 or inChannel != outChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannel),
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, numClasses=10) :
        super(ResNet,self).__init__()
        self.inChannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),   # 32*32*3 -> 32*32*64
            nn.BatchNorm2d(64),                                                 # 32*32*64 -> 32*32*64
            nn.ReLU(),                                                          # 32*32*64 -> 32*32*64
            nn.MaxPool2d(2, 2),                                                 # 32*32*64 -> 16*16*64
        )

        # self.model = nn.Sequential(
        #     self.conv1(),
        #     self.makeLayer(ResidualBlock, 64, 2, stride=1),                     # 16*16*64 -> 16*16*64
        #     self.makeLayer(ResidualBlock, 128, 2, stride=2),                    # 16*16*64 -> 8*8*128
        #     self.makeLayer(ResidualBlock, 256, 2, stride=2),                    # 8*8*128 -> 4*4*256
        #     self.makeLayer(ResidualBlock, 512, 2, stride=2),                    # 4*4*256 -> 2*2*512
        #     nn.AvgPool2d(4),                                                    # 2*2*512 -> 1*1*512
        #     Reshape(-1, 512),                                                   # 1*1*512 -> 512
        #     nn.Linear(512, numClasses),                                         # 512 -> 10
        #     nn.LogSoftmax(dim=1),                                               # 10 -> 10
        # )

        self.layer1 = self.makeLayer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.makeLayer(ResidualBlock, 128, 2, stride=1)
        self.layer3 = self.makeLayer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.makeLayer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, numClasses)

    def makeLayer(self, block, channels, numBlocks, stride):
        strides = [stride] + [1] * (numBlocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inChannel, channels, stride))
            self.inChannel = channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.conv1(x)                                 # 32*32*3 -> 16*16*64
        out = self.layer1(out)                              # 16*16*64 -> 16*16*64
        out = self.layer2(out)                              # 16*16*64 -> 8*8*128
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 8)               
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        out = functional.log_softmax(out,dim=1)
        return out

def CNN():
    return ResNet(ResidualBlock)


