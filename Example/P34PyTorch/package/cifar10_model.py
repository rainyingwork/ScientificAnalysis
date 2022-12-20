from torch import nn
import torch.nn.functional as F
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 ,10 ,3 ,1 ,1),  # 10*16*16
            nn.ReLU(),                  # 10*16*16
            nn.MaxPool2d(2 ,2),         # 10*8*8
            nn.Conv2d(10 ,20 ,3 ,1 ,1), # 20*8*8
            nn.ReLU(),                  # 20*8*8
            nn.MaxPool2d(2 ,2),         # 20*4*4
            nn.Conv2d(20 ,40 ,3 ,1 ,1), # 40*4*4
            nn.ReLU(),                  # 40*4*4
            nn.MaxPool2d(2, 2),         # 40*2*2
            Reshape(-1 ,40 *4 *4),      # 40*4*4
            nn.Dropout(0.2),            # 40*4*4
            nn.Linear(40*4*4,100) ,     # 100
            nn.ReLU(),                  # 100
            nn.Dropout(0.2),            # 100
            nn.Linear(100, 10),         # 10
        )

    def forward(self, x):
        w = self.model(x)
        op = F.log_softmax(w, dim=1)
        return op