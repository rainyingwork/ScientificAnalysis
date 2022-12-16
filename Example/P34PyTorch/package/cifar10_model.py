from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1=nn.Conv2d(3,10,3,1,1)
        self.conv2=nn.Conv2d(10,20,3,1,1)
        self.conv3=nn.Conv2d(20,40,3,1,1)
        self.pool=nn.MaxPool2d(2,2)
        self.linear1=nn.Linear(40*4*4,100)
        self.linear2=nn.Linear(100,10)
        self.dropout=nn.Dropout(0.2)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) # 10*16*16
        x=self.pool(F.relu(self.conv2(x))) # 20*8*8
        x=self.pool(F.relu(self.conv3(x))) # 40*4*4
        x=x.view(-1,40*4*4)
        x=self.dropout(x)
        x=F.relu(self.linear1(x))
        x=self.dropout(x)
        x=F.log_softmax(self.linear2(x),dim=1)
        return x