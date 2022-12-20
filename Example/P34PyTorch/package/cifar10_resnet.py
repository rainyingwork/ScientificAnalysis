import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1) :
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride !=1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel),
            )
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out =F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10) :
        super().__init__()
        self.inchannel=64
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.layer1=self.make_layer(ResidualBlock,64,2,stride=1)
        self.layer2=self.make_layer(ResidualBlock,128,2,stride=2)
        self.layer3=self.make_layer(ResidualBlock,256,2,stride=2)
        self.layer4=self.make_layer(ResidualBlock,512,2,stride=2)
        self.fc=nn.Linear(512,num_classes)
        self.fc2=nn.Linear(128*2*2,num_classes)
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides=[stride] + [1]*(num_blocks-1) # strides=[1,1]
        layers=[]
        for stride in strides:
            layers.append(block(self.inchannel,channels,stride))
            self.inchannel=channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.conv1(x)  # 64*16*16
        out=self.layer1(out) # 64*16*16
        out=self.layer2(out)  # 128*8*8      
        out=F.avg_pool2d(out,kernel_size=4)  # 128*2*2
        out=out.view(out.size(0),-1)       
        out=self.fc2(out)
        out=F.log_softmax(out,dim=1)
        return out

def CNN():
    return ResNet(ResidualBlock)


